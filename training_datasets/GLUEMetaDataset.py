from training_datasets.GLUEDataset import GLUEDataset
import learn2learn as l2l


class GLUEMetaDataset:
    def __init__(self, dataset: GLUEDataset, k, numTasks, filteredLabels=None):
        if filteredLabels is not None:
            self.glueDataset = l2l.data.FilteredMetaDataset(dataset, filteredLabels)
        else:
            self.glueDataset = l2l.data.MetaDataset(dataset)
        self.transforms = [
            l2l.data.transforms.NWays(self.glueDataset, n=len(set(self.glueDataset[:][1]))),
            l2l.data.transforms.KShots(self.glueDataset, k=k),  # sample all labels
            l2l.data.transforms.LoadData(self.glueDataset),
            l2l.data.transforms.RemapLabels(self.glueDataset),
            l2l.data.transforms.ConsecutiveLabels(self.glueDataset),
        ]
        self.taskSet = l2l.data.TaskDataset(dataset=self.glueDataset, task_transforms=self.transforms,
                                            num_tasks=numTasks)

    def getTask(self):
        return self.taskSet.sample()

    def getTotalTasks(self):
        return self.taskSet.num_tasks
