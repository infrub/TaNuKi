
class Tensor(TensorMixin):
    #methods for moving indices
    #I assumed that rollaxis is better than moveaxis in terms of computing costs
    #TODO pass if newIndices==oldIndices
    @outofplacable_tensorMixin_method
    def move_index_to_top(self, indexMoveFrom):
        indexMoveFrom = self.normarg_index(indexMoveFrom)
        labelMove = self.labels.pop(indexMoveFrom)
        self.labels.insert(0, labelMove)
        self.data = xp.rollaxis(self.data, indexMoveFrom, 0)

    @outofplacable_tensorMixin_method
    def move_index_to_bottom(self, indexMoveFrom):
        indexMoveFrom = self.normarg_index(indexMoveFrom)
        labelMove = self.labels.pop(indexMoveFrom)
        self.labels.append(labelMove)
        self.data = xp.rollaxis(self.data, indexMoveFrom, self.ndim)

    @outofplacable_tensorMixin_method
    def move_index_to_position(self, indexMoveFrom, position):
        indexMoveFrom = self.normarg_index(indexMoveFrom)
        labelMove = self.labels.pop(indexMoveFrom)
        self.labels.insert(position, labelMove)
        if position <= indexMoveFrom:
            self.data = xp.rollaxis(self.data, indexMoveFrom, position)
        else:
            self.data = xp.rollaxis(self.data, indexMoveFrom, position+1)

    @outofplacable_tensorMixin_method
    def move_indices_to_top(self, moveFrom):
        moveFrom = self.normarg_indices_front(moveFrom)
        moveTo = list(range(len(moveFrom)))
        notMoveFrom = diff_list(list(range(self.ndim)), moveFrom)
        newLabels = self.labels_of_indices(moveFrom) + self.labels_of_indices(notMoveFrom)
        self.data = xp.moveaxis(self.data, moveFrom, moveTo)
        self.labels = newLabels

    @outofplacable_tensorMixin_method
    def move_indices_to_bottom(self, moveFrom):
        moveFrom = self.normarg_indices_back(moveFrom)
        moveTo = list(range(self.ndim-len(moveFrom), self.ndim))
        notMoveFrom = diff_list(list(range(self.ndim)), moveFrom)
        newLabels = self.labels_of_indices(notMoveFrom) + self.labels_of_indices(moveFrom)
        self.data = xp.moveaxis(self.data, moveFrom, moveTo)
        self.labels = newLabels

    @outofplacable_tensorMixin_method
    def move_indices_to_position(self, moveFrom, position):
        moveFrom = self.normarg_indices_front(moveFrom)
        moveTo = list(range(position, position+len(moveFrom)))
        notMoveFrom = diff_list(list(range(self.ndim)), moveFrom)
        newLabels = self.labels_of_indices(notMoveFrom)
        newLabels = newLabels[:position] + self.labels_of_indices(moveFrom) + newLabels[position:]
        self.data = xp.moveaxis(self.data, moveFrom, moveTo)
        self.labels = newLabels

    @outofplacable_tensorMixin_method
    def move_all_indices(self, moveFrom):
        moveFrom = self.normarg_indices_front(moveFrom)
        moveTo = list(range(self.ndim))
        newLabels = self.labels_of_indices(moveFrom)

        oldPos_NewPos = [None for _ in range(self.ndim)]
        for i in range(self.ndim):
            oldPos_NewPos[moveTo[i]] = moveFrom[i]

        self.data = xp.transpose(self.data, oldPos_NewPos)
        self.labels = newLabels



    #methods for fuse/split
    #if new.. is no specified, assume like following:
    #["a","b","c","d"] <=split / fuse=> ["a",("b","c"),"d"]
    @outofplacable_tensorMixin_method
    def fuse_indices(self, splittedLabels=None, fusedLabel=None, memo=None):
        if memo is None:
            memo = {}

        if splittedLabels is None:
            if "splittedLabels" in memo:
                splittedLabels = memo["splittedLabels"]
            else:
                raise ValueError
        splittedIndices = self.normarg_indices(splittedLabels)
        splittedLabels = self.labels_of_indices(splittedIndices)

        if fusedLabel is None:
            if "fusedLabel" in memo:
                fusedLabel = memo["fusedLabel"]
            else:
                fusedLabel = tuple(splittedLabels)


        position = min(splittedIndices)
        self.move_indices_to_position(splittedIndices, position)
        del splittedIndices

        oldShape = self.shape
        splittedShape = oldShape[position:position+len(splittedLabels)]
        fusedDim = soujou(splittedShape)
        newShape = oldShape[:position] + (fusedDim,) + oldShape[position+len(splittedLabels):]

        oldLabels = self.labels
        newLabels = oldLabels[:position] + [fusedLabel] + oldLabels[position+len(splittedLabels):]

        self.data = xp.reshape(self.data, newShape)
        self.labels = newLabels

        memo.update({"splittedShape":splittedShape, "splittedLabels":splittedLabels, "fusedDim":fusedDim, "fusedLabel":fusedLabel})
        return memo #if out-of-place not returned. if you want, prepare a dict as memo in argument


    @outofplacable_tensorMixin_method
    def split_index(self, fusedLabel=None, splittedShape=None, splittedLabels=None, memo=None):
        if memo is None:
            memo = {}

        if fusedLabel is None:
            if "fusedLabel" in memo:
                fusedLabel = memo["fusedLabel"]
            else:
                raise ValueError
        fusedIndex = self.normarg_index(fusedLabel)
        fusedLabel = self.label_of_index(fusedIndex)

        if splittedShape is None:
            if "splittedShape" in memo:
                splittedShape = memo["splittedShape"]
            else:
                raise ValueError
        splittedShape = tuple(splittedShape)

        if splittedLabels is None:
            if "splittedLabels" in memo:
                splittedLabels = memo["splittedLabels"]
            else:
                splittedLabels = list(fusedLabel)
        splittedLabels = normarg_labels(splittedLabels)

        
        assert len(splittedLabels) == len(splittedShape)

        fusedDim = self.dim(fusedIndex)
        position = fusedIndex
        del fusedIndex

        assert soujou(splittedShape) == fusedDim

        newShape = self.shape[:position] + splittedShape + self.shape[position+1:]
        newLabels = self.labels[:position] + splittedLabels + self.labels[position+1:]

        self.data = xp.reshape(self.data, newShape)
        self.labels = newLabels

        memo.update({"fusedDim":fusedDim, "fusedLabel":fusedLabel, "splittedShape":splittedShape, "splittedLabels":splittedLabels})
        return memo #if out-of-place not returned. if you want, prepare a dict as memo in argument



    @inplacable_tensorMixin_method
    def contract(self, *args, **kwargs):
        return contract(self, *args, **kwargs)













class DiagonalTensor(TensorMixin)
    """
    #methods for moving indices
    #hobo muimi
    @outofplacable_tensorMixin_method
    def move_index_to_top(self, labelMove):
        indexMoveFrom = self.index_of_label(labelMove)
        self.labels.pop(indexMoveFrom)
        self.labels.insert(0, labelMove)

    @outofplacable_tensorMixin_method
    def move_index_to_bottom(self, labelMove):
        indexMoveFrom = self.index_of_label(labelMove)
        self.labels.pop(indexMoveFrom)
        self.labels.append(labelMove)

    @outofplacable_tensorMixin_method
    def move_index_to_position(self, labelMove, position, inplace=True):
        indexMoveFrom = self.index_of_label(labelMove)
        self.labels.pop(indexMoveFrom)
        self.labels.insert(position, labelMove)

    @outofplacable_tensorMixin_method
    def move_indices_to_top(self, labelsMove):
        labelsMove = normarg_labels(labelsMove)
        self.labels = labelsMove + diff_list(self.labels, labelsMove)

    @outofplacable_tensorMixin_method
    def move_indices_to_bottom(self, labelsMove):
        labelsMove = normarg_labels(labelsMove)
        self.labels = diff_list(self.labels, labelsMove) + labelsMove

    @outofplacable_tensorMixin_method
    def move_indices_to_position(self, labelsMove, position):
        labelsMove = normarg_labels(labelsMove)
        labelsNotMove = diff_list(self.labels, labelsMove)
        self.labels = labelsNotMove[:position] + labelsMove + labelsNotMove[position:]

    @outofplacable_tensorMixin_method
    def move_all_indices(self, newLabels):
        newLabels = normarg_labels(newLabels)
        oldLabels = self.labels
        if sorted(newLabels) != sorted(oldLabels):
            raise ValueError(f"newLabels do not match oldLabels. oldLabels=={oldLabels}, newLabels=={newLabels}")
        self.labels = newLabels
    """
