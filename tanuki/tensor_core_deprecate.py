
class Tensor(TensorMixin):
    #methods for moving indices
    #I assumed that rollaxis is better than moveaxis in terms of computing costs
    #TODO pass if newIndices==oldIndices
    @inplacable_tensorMixin_method
    def move_index_to_top(self, indexMoveFrom):
        indexMoveFrom = self.normarg_index(indexMoveFrom)
        newData = xp.rollaxis(self.data, indexMoveFrom, 0)
        newLabels = self.labels[indexMoveFrom:indexMoveFrom+1] + self.labels[:indexMoveFrom] + self.labels[indexMoveFrom+1:]
        return Tensor(newData, newLabels)

    @inplacable_tensorMixin_method
    def move_index_to_bottom(self, indexMoveFrom):
        indexMoveFrom = self.normarg_index(indexMoveFrom)
        newData = xp.rollaxis(self.data, indexMoveFrom, self.ndim)
        newLabels = self.labels[:indexMoveFrom] + self.labels[indexMoveFrom+1:] + self.labels[indexMoveFrom:indexMoveFrom+1]
        return Tensor(newData, newLabels)

    @inplacable_tensorMixin_method
    def move_index_to_position(self, indexMoveFrom, position):
        indexMoveFrom = self.normarg_index(indexMoveFrom)
        labelMove = self.labels.pop(indexMoveFrom)
        self.labels.insert(position, labelMove)
        if position <= indexMoveFrom:
            newData = xp.rollaxis(self.data, indexMoveFrom, position)
            newLabels = self.labels[:position] + self.labels[indexMoveFrom:indexMoveFrom+1] + self.labels[position:indexMoveFrom] + self.labels[indexMoveFrom+1:]
        else:
            newData = xp.rollaxis(self.data, indexMoveFrom, position+1)
            newLabels = self.labels[:indexMoveFrom] + self.labels[indexMoveFrom+1:position+1] + self.labels[indexMoveFrom:indexMoveFrom+1] + self.labels[position+1:]
        return Tensor(newData, newLabels)

    @inplacable_tensorMixin_method
    def move_indices_to_top(self, moveFrom):
        moveFrom = self.normarg_indices_front(moveFrom)
        moveTo = list(range(len(moveFrom)))
        notMoveFrom = diff_list(list(range(self.ndim)), moveFrom)
        newLabels = self.labels_of_indices(moveFrom) + self.labels_of_indices(notMoveFrom)
        newData = xp.moveaxis(self.data, moveFrom, moveTo)
        return Tensor(newData, newLabels)

    @inplacable_tensorMixin_method
    def move_indices_to_bottom(self, moveFrom):
        moveFrom = self.normarg_indices_back(moveFrom)
        moveTo = list(range(self.ndim-len(moveFrom), self.ndim))
        notMoveFrom = diff_list(list(range(self.ndim)), moveFrom)
        newLabels = self.labels_of_indices(notMoveFrom) + self.labels_of_indices(moveFrom)
        newData = xp.moveaxis(self.data, moveFrom, moveTo)
        return Tensor(newData, newLabels)

    @inplacable_tensorMixin_method
    def move_indices_to_position(self, moveFrom, position):
        moveFrom = self.normarg_indices(moveFrom)
        moveTo = list(range(position, position+len(moveFrom)))
        notMoveFrom = diff_list(list(range(self.ndim)), moveFrom)
        newLabels = self.labels_of_indices(notMoveFrom)
        newLabels = newLabels[:position] + self.labels_of_indices(moveFrom) + newLabels[position:]
        newData = xp.moveaxis(self.data, moveFrom, moveTo)
        return Tensor(newData, newLabels)

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
