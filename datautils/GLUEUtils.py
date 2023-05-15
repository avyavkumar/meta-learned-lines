from datasets import load_dataset

def list_glue_tasks():
    return ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', 'mnli_mismatched', 'mnli_matched', 'qnli', 'rte', 'wnli', 'ax']

def get_glue_task(task):
    return load_dataset('glue', task)

def get_glue_tasks():
    list_tasks = list_glue_tasks()
    glue_datasets = [get_glue_task(list_tasks[i]) for i in range(len(list_tasks))]
    return glue_datasets
