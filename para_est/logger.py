
class Logger():
    """
    Logger for counting function evaluations
    """

    def __init__(self,):
        self.f_eval_count = 0

    def increment(self):
        self.f_eval_count += 1

    def reset(self):
        self.f_eval_count = 0

    def get_f_eval_count(self):
        return self.f_eval_count