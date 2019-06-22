
    #methods for dummy index
    @inplacable_tensorMixin_method
    def add_dummy_index(self, label=()):
        newData = self.data[xp.newaxis, :]
        newLabels = [label] + self.labels
        return Tensor(newData, newLabels)

    @inplacable_tensorMixin_method
    def remove_dummy_indices(self, indices):
        indices = self.normarg_indices(indices)
        oldShape = self.shape
        oldLabels = self.labels
        newShape = ()
        newLabels = []
        for i, x in enumerate(oldLabels):
            if oldShape[i]==1 and i in indices:
                pass
            else:
                newShape = newShape + (oldShape[i],)
                newLabels.append(x)
        newData = self.data.reshape(newShape)
        return Tensor(newData, newLabels)

