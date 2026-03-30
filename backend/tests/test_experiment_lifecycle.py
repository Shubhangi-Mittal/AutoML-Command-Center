import os
import tempfile
import unittest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models import dataset, experiment, job  # noqa: F401
from app.models.dataset import Dataset
from app.models.experiment import Experiment
from app.models.job import TrainingJob
from app.services.experiment_tracker import ExperimentTracker
from app.tasks.training_tasks import _finalize_experiment_if_ready


class ExperimentLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        db_path = os.path.join(self.temp_dir.name, "test.db")
        engine = create_engine(f"sqlite:///{db_path}")
        session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        Base.metadata.create_all(bind=engine)
        self.db = session_factory()

        self.dataset = Dataset(
            name="sample.csv",
            file_path="/tmp/sample.csv",
            rows=10,
            columns=3,
            target_column="target",
            task_type="classification",
        )
        self.db.add(self.dataset)
        self.db.commit()
        self.db.refresh(self.dataset)

    def tearDown(self):
        self.db.close()
        self.temp_dir.cleanup()

    def _create_experiment(self, optimization_metric: str = "f1") -> Experiment:
        tracker = ExperimentTracker(self.db)
        return tracker.create_experiment(
            dataset_id=self.dataset.id,
            name="Experiment",
            optimization_metric=optimization_metric,
        )

    def test_finalize_experiment_waits_for_all_jobs(self):
        experiment = self._create_experiment()
        completed_job = TrainingJob(
            dataset_id=self.dataset.id,
            experiment_id=experiment.id,
            status="completed",
            model_type="linear",
            metrics={"f1": 0.72},
        )
        running_job = TrainingJob(
            dataset_id=self.dataset.id,
            experiment_id=experiment.id,
            status="running",
            model_type="xgboost",
        )
        self.db.add_all([completed_job, running_job])
        self.db.commit()

        _finalize_experiment_if_ready(self.db, experiment.id)
        refreshed = self.db.query(Experiment).filter(Experiment.id == experiment.id).first()

        self.assertEqual(refreshed.status, "running")
        self.assertIsNone(refreshed.best_job_id)

    def test_finalize_experiment_completes_when_jobs_are_terminal(self):
        experiment = self._create_experiment()
        weaker_job = TrainingJob(
            dataset_id=self.dataset.id,
            experiment_id=experiment.id,
            status="completed",
            model_type="linear",
            metrics={"f1": 0.61},
        )
        stronger_job = TrainingJob(
            dataset_id=self.dataset.id,
            experiment_id=experiment.id,
            status="completed",
            model_type="xgboost",
            metrics={"f1": 0.84},
        )
        failed_job = TrainingJob(
            dataset_id=self.dataset.id,
            experiment_id=experiment.id,
            status="failed",
            model_type="random_forest",
        )
        self.db.add_all([weaker_job, stronger_job, failed_job])
        self.db.commit()
        self.db.refresh(stronger_job)

        _finalize_experiment_if_ready(self.db, experiment.id)
        refreshed = self.db.query(Experiment).filter(Experiment.id == experiment.id).first()

        self.assertEqual(refreshed.status, "completed")
        self.assertEqual(refreshed.best_job_id, stronger_job.id)

    def test_experiment_tracker_uses_lower_is_better_for_rmse(self):
        experiment = self._create_experiment(optimization_metric="rmse")
        higher_rmse = TrainingJob(
            dataset_id=self.dataset.id,
            experiment_id=experiment.id,
            status="completed",
            model_type="linear",
            metrics={"rmse": 4.2},
        )
        lower_rmse = TrainingJob(
            dataset_id=self.dataset.id,
            experiment_id=experiment.id,
            status="completed",
            model_type="random_forest",
            metrics={"rmse": 1.8},
        )
        self.db.add_all([higher_rmse, lower_rmse])
        self.db.commit()
        self.db.refresh(lower_rmse)

        tracker = ExperimentTracker(self.db)
        completed = tracker.complete_experiment(experiment.id)

        self.assertEqual(completed.best_job_id, lower_rmse.id)


if __name__ == "__main__":
    unittest.main()
