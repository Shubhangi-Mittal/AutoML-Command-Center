export interface Dataset {
  id: string;
  name: string;
  rows: number | null;
  columns: number | null;
  size_bytes: number | null;
  profile: DataProfile | null;
  target_column: string | null;
  task_type: string | null;
  created_at: string;
}

export interface DataProfile {
  row_count: number;
  column_count: number;
  memory_usage_mb: number;
  duplicate_rows: number;
  columns: Record<string, ColumnProfile>;
  correlations: { top_pairs: CorrelationPair[] };
  warnings: string[];
  suggested_target: string | null;
  suggested_task_type: string | null;
}

export interface ColumnProfile {
  dtype: string;
  dtype_category: "numeric" | "categorical";
  missing_count: number;
  missing_pct: number;
  unique_count: number;
  unique_pct: number;
  mean?: number;
  median?: number;
  std?: number;
  min?: number;
  max?: number;
  q25?: number;
  q75?: number;
  skewness?: number;
  kurtosis?: number;
  zero_count?: number;
  negative_count?: number;
  histogram?: HistogramBin[];
  top_values?: Record<string, number>;
  avg_length?: number;
}

export interface HistogramBin {
  bin_start: number;
  bin_end: number;
  count: number;
}

export interface CorrelationPair {
  col1: string;
  col2: string;
  correlation: number;
}

export interface TrainingJob {
  id: string;
  dataset_id: string;
  experiment_id?: string;
  status: "pending" | "running" | "completed" | "failed";
  model_type: string;
  hyperparameters?: Record<string, any>;
  metrics?: Record<string, number>;
  feature_importance?: Record<string, number>;
  training_duration_seconds?: number;
  error_message?: string;
  is_best?: boolean;
  created_at: string;
  completed_at?: string;
}

export interface Experiment {
  id: string;
  dataset_id: string;
  name?: string;
  status: string;
  optimization_metric?: string;
  best_job_id?: string;
  created_at: string;
}

export interface TrainResult {
  dataset_id: string;
  experiment_id: string;
  task_type: string;
  target_column: string;
  best_job_id: string;
  feature_engineering: {
    transformations: string[];
    feature_count: number;
    feature_names: string[];
    train_size: number;
    test_size: number;
  };
  jobs: {
    job_id: string;
    model_type: string;
    metrics: Record<string, number>;
    feature_importance?: Record<string, number>;
    training_duration_seconds: number;
    model_path?: string;
  }[];
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  tool_calls?: { tool: string; input: any; output_summary?: string }[];
  timestamp: Date;
}

export interface ServingStatus {
  status: string;
  job_id?: string;
  model_type?: string;
  metrics?: Record<string, number>;
  deployed_at?: string;
}
