from datasets import load_dataset


def list_glue_tasks():
    return ['cola', 'sst2', 'mrpc', 'qqp', 'mnli', 'qnli', 'rte', 'snli']


def get_glue_task(task):
    if task == 'snli':
        return load_dataset(task)
    else:
        return load_dataset('glue', task)


def get_glue_tasks():
    list_tasks = list_glue_tasks()
    glue_datasets = [get_glue_task(list_tasks[i]) for i in range(len(list_tasks))]
    return glue_datasets


def get_entailment_tasks():
    return ['mnli', 'rte', 'qnli', 'snli']


def get_classification_tasks():
    return ['cola', 'sst2']


def get_equivalence_tasks():
    return ['qqp', 'mrpc']
