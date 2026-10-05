import lm_eval
from lm_eval.tasks import TaskManager

tm = TaskManager()
task_idx = tm.load_task_or_group(["wikitext"])
task = task_idx["wikitext"]
print("dataset:", task.config.dataset_path, task.config.dataset_name,
      "| test split:", task.config.test_split)
docs = list(task.test_docs()) if task.has_test_docs() else list(task.validation_docs())
print("num docs:", len(docs))
lens = [len(d) for d in docs]
print("doc lens chars: min", min(lens), "max", max(lens), "sum", sum(lens))
