"""AI Agent with fast-path routing and tool-calling backends."""

import json
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from app.config import settings
from app.agent_tools.tools import TOOL_DEFINITIONS, TOOL_EXECUTORS


SYSTEM_PROMPT = """You are an expert ML engineer assistant powering the AutoML Command Center.
You help users profile datasets, engineer features, train models, compare experiments, deploy models, and make predictions.

You have access to these tools:
- profile_dataset: Analyze a dataset's columns, statistics, correlations, and quality warnings.
- get_dataset_sample: View sample rows from the dataset.
- launch_training: Train models (linear, xgboost, random_forest) with automatic feature engineering.
- query_experiments: Compare model results sorted by optimization metric.
- deploy_model: Deploy the best model to the prediction API.
- get_prediction_template: Generate sample JSON input for making predictions. Use when users ask for test/sample/example JSON.
- make_prediction: Make a prediction with the deployed model. Use when users want to test or try the model.
- get_serving_status: Check if a model is currently deployed and its details.
- suggest_improvements: Analyze results and suggest ways to improve model performance.

Workflow:
1. When a user asks to analyze/profile data, call profile_dataset first.
2. When asked to train models, call launch_training. You can specify target_column, task_type, model_types, and optimization_metric.
3. After training, summarize the results clearly: which model won, key metrics, and top features.
4. When asked to deploy, call deploy_model with the best job_id.
5. When asked for sample/test JSON or to test a prediction: first call get_prediction_template to get sample data, then call make_prediction with those features. Show the user both the input JSON and the prediction result.
6. When asked how to improve, call suggest_improvements.
7. Be concise but informative. Use numbers and percentages.

If the user provides a dataset_id, use it. Otherwise, ask which dataset to work with.
"""


def _claude_tools_to_openai(claude_tools: list) -> list:
    """Convert Claude tool definitions to OpenAI/Groq format."""
    openai_tools = []
    for tool in claude_tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"],
            },
        })
    return openai_tools


OPENAI_TOOL_DEFINITIONS = _claude_tools_to_openai(TOOL_DEFINITIONS)
FAST_PATH_KEYWORDS = {
    "predict": ["predict", "test json", "sample json", "example json", "try the model", "test prediction"],
    "improve": ["improve", "suggestion", "better", "how can i"],
    "status": ["status", "deployed", "serving"],
    "deploy": ["deploy", "serve", "production"],
    "profile": ["profile", "analyze", "describe", "explore", "look at"],
    "train": ["train", "build", "fit", "learn"],
    "compare": ["compare", "experiment", "result", "which"],
    "sample": ["sample", "show", "preview", "head"],
}
MAX_HISTORY_MESSAGES = 12


class MLAgent:
    """ReAct-style ML agent using Claude/Groq API with tool calling."""

    def __init__(self):
        self.conversations: Dict[str, List[Dict]] = {}
        self._groq_client = None
        self._anthropic_client = None

    async def chat(
        self,
        user_message: str,
        session_id: str,
        dataset_id: Optional[str],
        db: Session,
    ) -> Dict[str, Any]:
        """Process a user message and return the agent's response."""
        if self._should_use_fast_path(user_message):
            return await self._chat_fallback(user_message, session_id, dataset_id, db)

        if settings.GROQ_API_KEY:
            return await self._chat_with_groq(user_message, session_id, dataset_id, db)
        elif settings.ANTHROPIC_API_KEY:
            return await self._chat_with_claude(user_message, session_id, dataset_id, db)
        else:
            return await self._chat_fallback(user_message, session_id, dataset_id, db)

    async def _chat_with_groq(
        self, user_message: str, session_id: str, dataset_id: Optional[str], db: Session,
    ) -> Dict[str, Any]:
        """Full ReAct loop using Groq API with tool use."""
        from groq import Groq

        client = self._get_groq_client(Groq)

        history = self._get_history(session_id)
        initial_history_length = len(history)
        history.append({"role": "user", "content": self._build_user_content(user_message, dataset_id)})

        tool_calls_summary = []
        max_iterations = 10

        for _ in range(max_iterations):
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    tools=OPENAI_TOOL_DEFINITIONS,
                    tool_choice="auto",
                    max_tokens=4096,
                )
            except Exception as exc:
                history[:] = history[:initial_history_length]

                if settings.ANTHROPIC_API_KEY:
                    return await self._chat_with_claude(user_message, session_id, dataset_id, db)

                if self._is_groq_tool_failure(exc):
                    return await self._chat_fallback(user_message, session_id, dataset_id, db)

                return {
                    "response": (
                        "The Groq agent hit a provider-side tool-calling error, so I switched to the local workflow. "
                        "Please try the request again if you'd like."
                    ),
                    "tool_calls": [],
                }

            message = response.choices[0].message

            # Add assistant message to history
            history.append(message.model_dump(exclude_none=True))

            if message.tool_calls:
                for tc in message.tool_calls:
                    tool_name = tc.function.name
                    tool_input = self._parse_tool_arguments(tc.function.arguments)
                    result, summary = await self._execute_tool(tool_name, tool_input, dataset_id, db)
                    tool_calls_summary.append(summary)

                    history.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, default=str),
                    })
            else:
                final_text = message.content or ""
                self._trim_history(history)

                return {
                    "response": final_text,
                    "tool_calls": tool_calls_summary,
                }

        return {
            "response": "I've reached the maximum number of steps. Please try a simpler request.",
            "tool_calls": tool_calls_summary,
        }

    async def _chat_with_claude(
        self, user_message: str, session_id: str, dataset_id: Optional[str], db: Session,
    ) -> Dict[str, Any]:
        """Full ReAct loop using Claude API with tool use."""
        import anthropic

        client = self._get_anthropic_client(anthropic)
        history = self._get_history(session_id)
        history.append({"role": "user", "content": self._build_user_content(user_message, dataset_id)})

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
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        tool_name = block.name
                        result, summary = await self._execute_tool(tool_name, block.input, dataset_id, db)
                        tool_calls.append(summary)

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

                self._trim_history(history)

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
        if any(w in message_lower for w in ["predict", "test json", "sample json", "example json", "try the model", "test prediction"]):
            # Get prediction template and make a prediction
            if not dataset_id:
                return {"response": "Please select a dataset first.", "tool_calls": []}

            template = await TOOL_EXECUTORS["get_prediction_template"](dataset_id=dataset_id, db=db)
            tool_calls.append({"tool": "get_prediction_template", "input": {"dataset_id": dataset_id}, "output_summary": _summarize_result(template)})

            if "error" in template:
                return {"response": template["error"], "tool_calls": tool_calls}

            # Try making a prediction with the sample
            pred_result = await TOOL_EXECUTORS["make_prediction"](features=template["sample_input"], db=db)
            tool_calls.append({"tool": "make_prediction", "input": template["sample_input"], "output_summary": _summarize_result(pred_result)})

            response = f"**Sample Prediction for {template['dataset_name']}**\n\n"
            response += f"Target column: `{template['target_column']}` ({template['task_type']})\n\n"
            response += f"**Input JSON:**\n```json\n{json.dumps(template['sample_input'], indent=2, default=str)}\n```\n\n"

            if "error" in pred_result:
                response += f"**Prediction failed:** {pred_result['error']}\n\nDeploy a model first, then try again."
            else:
                response += f"**Prediction:** `{template['target_column']}` = **{pred_result['prediction']}**\n"
                if pred_result.get("probabilities"):
                    response += f"**Probabilities:** {pred_result['probabilities']}\n"
                response += f"\nModel: {pred_result.get('model_type', 'N/A')}"

            return {"response": response, "tool_calls": tool_calls}

        elif any(w in message_lower for w in ["improve", "suggestion", "better", "how can i"]):
            if not dataset_id:
                return {"response": "Please select a dataset first.", "tool_calls": []}

            result = await TOOL_EXECUTORS["suggest_improvements"](dataset_id=dataset_id, db=db)
            tool_calls.append({"tool": "suggest_improvements", "input": {"dataset_id": dataset_id}, "output_summary": _summarize_result(result)})

            if "error" in result:
                return {"response": result["error"], "tool_calls": tool_calls}

            response = f"**Improvement Suggestions for {result['dataset_name']}**\n\n"
            response += f"Best model: **{result['best_model']}** ({result['models_trained']} models trained)\n\n"
            for i, s in enumerate(result["suggestions"], 1):
                response += f"{i}. {s}\n"
            return {"response": response, "tool_calls": tool_calls}

        elif any(w in message_lower for w in ["status", "deployed", "serving"]):
            result = await TOOL_EXECUTORS["get_serving_status"](db=db)
            tool_calls.append({"tool": "get_serving_status", "input": {}, "output_summary": _summarize_result(result)})

            if result["status"] == "deployed":
                response = (
                    f"**Model is deployed** ✅\n\n"
                    f"- **Type:** {result.get('model_type', 'N/A')}\n"
                    f"- **Dataset:** {result.get('dataset_name', 'N/A')}\n"
                    f"- **Deployed at:** {result.get('deployed_at', 'N/A')}\n"
                )
            else:
                response = "**No model deployed.** Train models and deploy the best one first."
            return {"response": response, "tool_calls": tool_calls}

        elif any(w in message_lower for w in ["deploy", "serve", "production"]):
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
                    "- **\"Deploy the best model\"** — serve predictions via API\n"
                    "- **\"Give me sample test JSON\"** — get a prediction template + result\n"
                    "- **\"How can I improve?\"** — get suggestions for better performance\n"
                    "- **\"What's the serving status?\"** — check deployed model info\n\n"
                    "Select a dataset first, then tell me what to do!"
                ),
                "tool_calls": [],
            }

    def reset(self, session_id: str) -> None:
        """Clear conversation history for a session."""
        self.conversations.pop(session_id, None)

    def _should_use_fast_path(self, user_message: str) -> bool:
        message_lower = user_message.lower()
        matched_categories = [
            category
            for category, keywords in FAST_PATH_KEYWORDS.items()
            if any(keyword in message_lower for keyword in keywords)
        ]
        return len(matched_categories) == 1

    def _get_history(self, session_id: str) -> List[Dict[str, Any]]:
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        return self.conversations[session_id]

    def _build_user_content(self, user_message: str, dataset_id: Optional[str]) -> str:
        if not dataset_id:
            return user_message
        return f"[Active dataset ID: {dataset_id}]\n\n{user_message}"

    def _get_groq_client(self, groq_client_cls):
        if self._groq_client is None:
            self._groq_client = groq_client_cls(api_key=settings.GROQ_API_KEY)
        return self._groq_client

    def _get_anthropic_client(self, anthropic_module):
        if self._anthropic_client is None:
            self._anthropic_client = anthropic_module.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        return self._anthropic_client

    def _parse_tool_arguments(self, raw_arguments: Optional[str]) -> Dict[str, Any]:
        if not raw_arguments:
            return {}
        try:
            return json.loads(raw_arguments)
        except json.JSONDecodeError:
            return {}

    async def _execute_tool(
        self,
        tool_name: str,
        tool_input: Optional[Dict[str, Any]],
        dataset_id: Optional[str],
        db: Session,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        resolved_input = dict(tool_input or {})
        if dataset_id and "dataset_id" not in resolved_input:
            resolved_input["dataset_id"] = dataset_id

        executor = TOOL_EXECUTORS.get(tool_name)
        if executor:
            result = await executor(db=db, **resolved_input)
        else:
            result = {"error": f"Unknown tool: {tool_name}"}

        summary = {
            "tool": tool_name,
            "input": resolved_input,
            "output_summary": _summarize_result(result),
        }
        return result, summary

    def _trim_history(self, history: List[Dict[str, Any]]) -> None:
        if len(history) > MAX_HISTORY_MESSAGES:
            history[:] = history[-MAX_HISTORY_MESSAGES:]

    def _is_groq_tool_failure(self, exc: Exception) -> bool:
        message = str(exc).lower()
        return "tool_use_failed" in message or "failed to call a function" in message


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
