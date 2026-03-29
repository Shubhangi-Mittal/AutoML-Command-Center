"""AI Agent with ReAct pattern using Claude API, with intent-detection fallback."""

import json
import re
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from app.config import settings
from app.agent_tools.tools import TOOL_DEFINITIONS, TOOL_EXECUTORS


SYSTEM_PROMPT = """You are an expert ML engineer assistant powering the AutoML Command Center.
You help users profile datasets, engineer features, train models, compare experiments, and deploy the best model.

You have access to these tools:
- profile_dataset: Analyze a dataset's columns, statistics, correlations, and quality warnings.
- get_dataset_sample: View sample rows from the dataset.
- launch_training: Train models (linear, xgboost, random_forest) with automatic feature engineering.
- query_experiments: Compare model results sorted by optimization metric.
- deploy_model: Deploy the best model to the prediction API.

Workflow:
1. When a user asks to analyze/profile data, call profile_dataset first.
2. When asked to train models, call launch_training. You can specify target_column, task_type, model_types, and optimization_metric.
3. After training, summarize the results clearly: which model won, key metrics, and top features.
4. When asked to deploy, call deploy_model with the best job_id.
5. Be concise but informative. Use numbers and percentages.

If the user provides a dataset_id, use it. Otherwise, ask which dataset to work with.
"""


class MLAgent:
    """ReAct-style ML agent using Claude API with tool calling."""

    def __init__(self):
        self.conversations: Dict[str, List[Dict]] = {}

    async def chat(
        self,
        user_message: str,
        session_id: str,
        dataset_id: Optional[str],
        db: Session,
    ) -> Dict[str, Any]:
        """Process a user message and return the agent's response."""
        if settings.ANTHROPIC_API_KEY:
            return await self._chat_with_claude(user_message, session_id, dataset_id, db)
        else:
            return await self._chat_fallback(user_message, session_id, dataset_id, db)

    async def _chat_with_claude(
        self, user_message: str, session_id: str, dataset_id: Optional[str], db: Session,
    ) -> Dict[str, Any]:
        """Full ReAct loop using Claude API with tool use."""
        import anthropic

        client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

        # Get or create conversation history
        if session_id not in self.conversations:
            self.conversations[session_id] = []

        history = self.conversations[session_id]

        # Inject dataset context into user message
        if dataset_id:
            user_content = f"[Active dataset ID: {dataset_id}]\n\n{user_message}"
        else:
            user_content = user_message

        history.append({"role": "user", "content": user_content})

        tool_calls = []
        max_iterations = 10

        for _ in range(max_iterations):
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOL_DEFINITIONS,
                messages=history,
            )

            # Add assistant response to history
            history.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "tool_use":
                # Execute each tool call
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        tool_name = block.name
                        tool_input = block.input

                        # Inject dataset_id if not provided
                        if dataset_id and "dataset_id" not in tool_input:
                            tool_input["dataset_id"] = dataset_id

                        executor = TOOL_EXECUTORS.get(tool_name)
                        if executor:
                            result = await executor(db=db, **tool_input)
                        else:
                            result = {"error": f"Unknown tool: {tool_name}"}

                        tool_calls.append({
                            "tool": tool_name,
                            "input": tool_input,
                            "output_summary": _summarize_result(result),
                        })

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result, default=str),
                        })

                history.append({"role": "user", "content": tool_results})
            else:
                # Extract final text
                final_text = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        final_text += block.text

                # Keep conversation manageable
                if len(history) > 20:
                    history[:] = history[-16:]

                return {
                    "response": final_text,
                    "tool_calls": tool_calls,
                }

        return {
            "response": "I've reached the maximum number of steps. Please try a simpler request.",
            "tool_calls": tool_calls,
        }

    async def _chat_fallback(
        self, user_message: str, session_id: str, dataset_id: Optional[str], db: Session,
    ) -> Dict[str, Any]:
        """Intent-detection fallback when no API key is set."""
        message_lower = user_message.lower()
        tool_calls = []

        # Detect intent and call appropriate tools (order matters — more specific first)
        if any(w in message_lower for w in ["deploy", "serve", "production"]):
            result = await TOOL_EXECUTORS["deploy_model"](db=db, dataset_id=dataset_id)
            tool_calls.append({"tool": "deploy_model", "input": {"dataset_id": dataset_id}, "output_summary": _summarize_result(result)})

            if "error" in result:
                return {"response": result["error"], "tool_calls": tool_calls}

            response = (
                f"**Model deployed!** ✅\n\n"
                f"- **Type:** {result['model_type']}\n"
                f"- **Job:** `{result['job_id'][:8]}...`\n"
                f"- **Deployed at:** {result['deployed_at']}\n\n"
                f"You can now make predictions at `POST /api/serving/predict`"
            )
            return {"response": response, "tool_calls": tool_calls}

        elif any(w in message_lower for w in ["profile", "analyze", "describe", "explore", "look at"]):
            if not dataset_id:
                return {"response": "Please select a dataset first.", "tool_calls": []}

            result = await TOOL_EXECUTORS["profile_dataset"](dataset_id=dataset_id, db=db)
            tool_calls.append({"tool": "profile_dataset", "input": {"dataset_id": dataset_id}, "output_summary": _summarize_result(result)})

            if "error" in result:
                return {"response": result["error"], "tool_calls": tool_calls}

            response = (
                f"**Dataset: {result['dataset_name']}**\n\n"
                f"- **Rows:** {result['row_count']:,} | **Columns:** {result['column_count']} | **Memory:** {result['memory_usage_mb']} MB\n"
                f"- **Suggested target:** `{result['suggested_target']}` ({result['suggested_task_type']})\n"
                f"- **Duplicates:** {result['duplicate_rows']}\n"
            )
            if result["warnings"]:
                response += f"\n**Warnings:**\n" + "\n".join(f"- {w}" for w in result["warnings"])

            return {"response": response, "tool_calls": tool_calls}

        elif any(w in message_lower for w in ["train", "build", "model", "fit", "learn"]):
            if not dataset_id:
                return {"response": "Please select a dataset first.", "tool_calls": []}

            # Parse optimization metric from message
            opt_metric = None
            for metric in ["recall", "precision", "accuracy", "f1", "rmse", "mae", "r2"]:
                if metric in message_lower:
                    opt_metric = metric
                    break

            kwargs = {"dataset_id": dataset_id, "db": db}
            if opt_metric:
                kwargs["optimization_metric"] = opt_metric

            result = await TOOL_EXECUTORS["launch_training"](**kwargs)
            tool_calls.append({"tool": "launch_training", "input": {"dataset_id": dataset_id}, "output_summary": _summarize_result(result)})

            if "error" in result:
                return {"response": result["error"], "tool_calls": tool_calls}

            # Format results
            response = f"**Training complete!** Experiment: `{result['experiment_id'][:8]}...`\n\n"
            response += f"Feature engineering applied {len(result['feature_engineering']['transformations'])} transformations → {result['feature_engineering']['feature_count']} features\n\n"
            response += "| Model | " + " | ".join(result["jobs"][0]["metrics"].keys()) + " | Duration |\n"
            response += "|---|" + "|".join(["---"] * len(result["jobs"][0]["metrics"])) + "|---|\n"

            for job in result["jobs"]:
                is_best = "🏆 " if job["job_id"] == result["best_job_id"] else ""
                metrics_str = " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in job["metrics"].values())
                response += f"| {is_best}{job['model_type']} | {metrics_str} | {job['training_duration_seconds']:.3f}s |\n"

            # Top features from best model
            best_job = next((j for j in result["jobs"] if j["job_id"] == result["best_job_id"]), result["jobs"][0])
            if best_job.get("top_features"):
                response += f"\n**Top features ({best_job['model_type']}):** "
                response += ", ".join(f"`{k}` ({v:.3f})" for k, v in best_job["top_features"].items())

            response += "\n\nWant me to deploy the best model?"

            return {"response": response, "tool_calls": tool_calls}

        elif any(w in message_lower for w in ["compare", "experiment", "result", "which"]):
            result = await TOOL_EXECUTORS["query_experiments"](db=db, dataset_id=dataset_id)
            tool_calls.append({"tool": "query_experiments", "input": {"dataset_id": dataset_id}, "output_summary": _summarize_result(result)})

            if "error" in result:
                return {"response": result["error"], "tool_calls": tool_calls}

            response = f"**Experiment:** {result.get('name', 'N/A')} (optimizing: {result['optimization_metric']})\n\n"
            for job in result["jobs"]:
                best = "🏆 " if job["is_best"] else "   "
                metrics_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in job["metrics"].items() if k != "confusion_matrix")
                response += f"{best}**{job['model_type']}**: {metrics_str} ({job['training_duration_seconds']:.3f}s)\n"

            return {"response": response, "tool_calls": tool_calls}

        elif any(w in message_lower for w in ["sample", "show", "preview", "head"]):
            if not dataset_id:
                return {"response": "Please select a dataset first.", "tool_calls": []}

            result = await TOOL_EXECUTORS["get_dataset_sample"](dataset_id=dataset_id, db=db)
            tool_calls.append({"tool": "get_dataset_sample", "input": {"dataset_id": dataset_id}, "output_summary": _summarize_result(result)})

            if "error" in result:
                return {"response": result["error"], "tool_calls": tool_calls}

            response = f"**Sample from {result['dataset_name']}:**\n\n"
            response += "| " + " | ".join(result["columns"]) + " |\n"
            response += "|" + "|".join(["---"] * len(result["columns"])) + "|\n"
            for row in result["sample"]:
                response += "| " + " | ".join(str(row.get(c, ""))[:20] for c in result["columns"]) + " |\n"

            return {"response": response, "tool_calls": tool_calls}

        else:
            return {
                "response": (
                    "I can help you with:\n\n"
                    "- **\"Analyze this dataset\"** — profile columns, stats, warnings\n"
                    "- **\"Show me a sample\"** — preview rows from the data\n"
                    "- **\"Train models\"** — build and compare ML models\n"
                    "- **\"Train optimizing for recall\"** — specify a metric\n"
                    "- **\"Compare results\"** — see experiment comparisons\n"
                    "- **\"Deploy the best model\"** — serve predictions via API\n\n"
                    "Select a dataset first, then tell me what to do!"
                ),
                "tool_calls": [],
            }

    def reset(self, session_id: str) -> None:
        """Clear conversation history for a session."""
        self.conversations.pop(session_id, None)


def _summarize_result(result: Dict[str, Any]) -> str:
    """Create a brief summary of a tool result for display."""
    if "error" in result:
        return f"Error: {result['error']}"
    keys = list(result.keys())[:3]
    parts = []
    for k in keys:
        v = result[k]
        if isinstance(v, (str, int, float)):
            parts.append(f"{k}={v}")
    return ", ".join(parts) if parts else "OK"


# Singleton agent instance
_agent_instance: Optional[MLAgent] = None


def get_agent() -> MLAgent:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = MLAgent()
    return _agent_instance
