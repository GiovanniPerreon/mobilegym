from bench_env.task import BaseTask
from bench_env.task.judge import JudgeInput

class ReadHealthScore(BaseTask):
    description = (
        "Open the HealthMonitor app and make sure the displayed health score is 70 or higher. "
        "You can click the refresh button if the current score is too low."
    )
    apps = ["healthmonitor"]
    scope = "S1"
    objective = "read"
    composition = "atomic"
    difficulty = "L2"
    max_steps = 15
    capabilities = ["health"]
    expected_changes = ["apps.healthmonitor.healthScore"]

    def check_goals(self, input: JudgeInput) -> list[dict]:
        score = input.apps.get("healthmonitor", {}).get("healthScore")
        if score is None:
            return [{"passed": False, "reason": "Health score not found"}]
        if score >= 70:
            return [{"passed": True, "reason": f"Health score {score} is healthy (≥70)"}]
        else:
            return [{"passed": False, "reason": f"Health score {score} is below 70"}]

    def get_answer(self, input: JudgeInput) -> dict:
        return {"healthScore": input.apps.get("healthmonitor", {}).get("healthScore")}