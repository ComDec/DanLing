# pylint: disable=protected-access
from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Sequence, SupportsFloat

import torch
from torch import Tensor
from torch.utils.data._utils.collate import default_collate_fn_map

from ..utils import method_cache
from .utils import TorchFuncRegistry


def tensor_mask(
    tensors: Sequence[Tensor],
    size: torch.Size,
    *,
    batch_first: bool = True,
    padding_value: float = 0.0,
    mask_value: bool = False
) -> tuple[Tensor, Tensor]:
    r"""
    Build a padded tensor and corresponding tensor mask with a sequence of tensors and desired size.

    Args:
        tensors: sequence of tensors to be padded.
        size: desired size of the padded tensor (and mask tensor).
        batch_first: whether to put the batch dimension in the first dimension.
            Defaults to `True`.
        padding_value: padding value in the padded tensor.
            Defaults to `0.0`.
        mask_value: mask value in the mask tensor.
            Defaults to `False`.

    Returns:
        (tuple[Tensor, Tensor]): padded tensor and corresponding tensor mask

    Examples:
        >>> tensor_list = [torch.tensor([1, 2, 3]), torch.tensor([4, 5])]
        >>> tensor, mask = tensor_mask(tensor_list, (2, 3))
        >>> tensor
        tensor([[1, 2, 3],
                [4, 5, 0]])
        >>> mask
        tensor([[ True,  True,  True],
                [ True,  True, False]])
    """

    tensor = torch.full(size, fill_value=padding_value, dtype=tensors[0].dtype, device=tensors[0].device)
    mask = torch.full(size, fill_value=mask_value, dtype=torch.bool, device=tensors[0].device)
    for i, t in enumerate(tensors):
        tensor[i][tuple(slice(0, t.shape[dim]) for dim in range(len(size) - 1))] = t  # type: ignore
        mask[i][tuple(slice(0, t.shape[dim]) for dim in range(len(size) - 1))] = not mask_value
    if not batch_first:
        tensor, mask = tensor.transpose(0, 1), mask.transpose(0, 1)
    return tensor, mask


def pad_tensor(
    tensors: Sequence[Tensor], size: torch.Size, *, batch_first: bool = True, padding_value: float = 0.0
) -> Tensor:
    r"""
    Pads a tensor with a sequence of tensors and desired size.

    Args:
        tensors: sequence of tensors to be padded.
        size: desired size of the padded tensor (and mask tensor).
        batch_first: whether to put the batch dimension in the first dimension.
            Defaults to `True`.
        mask_value: mask value in the mask tensor.
            Defaults to `False`.

    Returns:
        (Tensor): padded tensor
    """

    ret = torch.full(size, fill_value=padding_value, dtype=tensors[0].dtype, device=tensors[0].device)
    for i, t in enumerate(tensors):
        ret[i][tuple(slice(0, t.shape[dim]) for dim in range(len(size) - 1))] = t  # type: ignore
    if not batch_first:
        ret = ret.transpose(0, 1)
    return ret


def mask_tensor(
    tensors: Sequence[Tensor], size: torch.Size, *, batch_first: bool = True, mask_value: bool = False
) -> Tensor:
    r"""
    Build a tensor mask with a sequence of tensors and desired size.

    Args:
        tensors: sequence of tensors to be padded.
        size: desired size of the padded tensor (and mask tensor).
        batch_first: whether to put the batch dimension in the first dimension.
            Defaults to `True`.
        mask_value: mask value in the mask tensor.
            Defaults to `False`.

    Returns:
        (Tensor): tensor mask
    """

    ret = torch.full(size, fill_value=mask_value, dtype=torch.bool, device=tensors[0].device)
    for i, t in enumerate(tensors):
        ret[i][tuple(slice(0, t.shape[dim]) for dim in range(len(size) - 1))] = not mask_value
    if not batch_first:
        ret = ret.transpose(0, 1)
    return ret


class PNTensor(Tensor):
    r"""
    Wrapper for tensors to be converted to `NestedTensor`.

    `PNTensor` is a subclass of `torch.Tensor`.
    It implements two additional methods as `NestedTensor`: `tensor` and `mask`.

    Although it is possible to construct `NestedTensor` in dataset,
    the best practice is to do so in `collate_fn`.
    However, it is hard to tell if a batch of `Tensor` should be stacked or converted to `NestedTensor`.

    `PNTensor` is introduced overcome this limitation.

    Convert tensors that will be converted to `NestedTensor` to a `PNTensor`,
    and all you need to do is to convert `PNTensor` to `NestedTensor` in `collate_fn`.
    """

    @property
    def tensor(self) -> Tensor:
        r"""
        Identical to `self`.

        Returns:
            (torch.Tensor):

        Examples:
            >>> tensor = torch.tensor([1, 2, 3])
            >>> pn_tensor = PNTensor(tensor)
            >>> (tensor == pn_tensor).all()
            PNTensor(True)
            >>> (tensor == pn_tensor.tensor).all()
            PNTensor(True)
        """

        return self

    @property
    def mask(self) -> Tensor:
        r"""
        Identical to `torch.ones_like(self)`.

        Returns:
            (torch.Tensor):

        Examples:
            >>> tensor = torch.tensor([1, 2, 3])
            >>> pn_tensor = PNTensor(tensor)
            >>> (pn_tensor.mask == torch.ones_like(pn_tensor)).all()
            PNTensor(True)
        """

        return torch.ones_like(self)

    def new_empty(self, *args, **kwargs):
        return PNTensor(super().new_empty(*args, **kwargs))


class NestedTensor:
    r"""
    Wrap an iterable of tensors into a single tensor with a mask.

    In sequence to sequence tasks, elements of a batch are usually not of the same length.
    This made it tricky to use a single tensor to represent a batch of sequences.

    `NestedTensor` allows to store a sequence of tensors of different lengths in a single object.
    It also provides a mask that can be used to retrieve the original sequence of tensors.

    When calling `__getitem__(arg)` on a `NestedTensor`, it has two return type:
    1. if arg is `int` or `slice`, returns a tuple of two `tensor`s, representing data and padding mask.
    2. if arg is a `tuple`, return a new `NestedTensor` with specified shape.

    Attributes:
        _storage: The sequence of tensors.
        tensor: padded tensor.
        mask: mask tensor.
        batch_first:  Whether the first dimension of the tensors is the batch dimension.

            If `True`, the first dimension is the batch dimension, i.e., `B, N, *`.

            If `False`, the first dimension is the sequence dimension, i.e., `N, B, *`
        padding_value: The padding value used to in padded tensor.
        mask_value: The mask value used in mask tensor.

    Args:
        tensors:
        batch_first:
        padding_value:
        mask_value:

    Raises:
        ValueError: If `tensors` is not an iterable.
        ValueError: If `tensors` is empty.

    Notes:
        We have rewritten the `__getattr__` function to support as much native tensor operations as possible.
        However, not all operations are tested.

        Please file an issue if you find any bugs.

    Examples:
        >>> nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
        >>> nested_tensor.shape
        torch.Size([2, 3])
        >>> nested_tensor.device
        device(type='cpu')
        >>> nested_tensor.dtype
        torch.int64
        >>> nested_tensor.tensor
        tensor([[1, 2, 3],
                [4, 5, 0]])
        >>> nested_tensor.mask
        tensor([[ True,  True,  True],
                [ True,  True, False]])
        >>> nested_tensor.to(torch.float).tensor
        tensor([[1., 2., 3.],
                [4., 5., 0.]])
        >>> nested_tensor.half().tensor
        tensor([[1., 2., 3.],
                [4., 5., 0.]], dtype=torch.float16)
        >>> nested_tensor[:]
        (tensor([[1, 2, 3],
                [4, 5, 0]]), tensor([[ True,  True,  True],
                [ True,  True, False]]))
        >>> nested_tensor[1]
        (tensor([4, 5]), tensor([True, True]))
        >>> nested_tensor[:, 1:]
        NestedTensor([[2, 3],
                [5, 0]])
        >>> nested_tensor.tolist()
        [[1, 2, 3], [4, 5]]
        >>> NestedTensor(*[[1, 2, 3], [4, 5]])
        NestedTensor([[1, 2, 3],
                [4, 5, 0]])
        >>> NestedTensor()
        Traceback (most recent call last):
        ValueError: NestedTensor must be initialised with a non-empty Iterable.
        >>> NestedTensor(False)
        Traceback (most recent call last):
        ValueError: NestedTensor must be initialised with an Iterable, bug got <class 'bool'>.
    """

    _storage: Sequence[Tensor]
    batch_first: bool = True
    padding_value: SupportsFloat = 0.0
    mask_value: bool = False

    def __init__(
        self,
        *tensors: Iterable[Tensor],
        batch_first: bool = True,
        padding_value: SupportsFloat = 0.0,
        mask_value: bool = False,
    ) -> None:
        if len(tensors) == 1 and isinstance(tensors, Sequence):
            tensors = tensors[0]  # type: ignore
        if not isinstance(tensors, Iterable):
            raise ValueError(f"NestedTensor must be initialised with an Iterable, bug got {type(tensors)}.")
        tensors = list(tensors)  # type: ignore
        if len(tensors) == 0:
            raise ValueError("NestedTensor must be initialised with a non-empty Iterable.")
        if not isinstance(tensors[0], Tensor):
            tensors = [torch.tensor(tensor) for tensor in tensors]  # type: ignore
        self._storage = tensors
        self.batch_first = batch_first
        self.padding_value = padding_value
        self.mask_value = mask_value

    def storage(self):
        return self._storage

    @property
    def tensor(self) -> Tensor:
        r"""
        Return a single tensor by padding all the tensors.

        Returns:
            (torch.Tensor):

        Examples:
            >>> nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
            >>> nested_tensor.tensor
            tensor([[1, 2, 3],
                    [4, 5, 0]])
        """

        return self._tensor(tuple(self._storage))

    @property
    def mask(self) -> Tensor:
        r"""
        Padding mask of `tensor`.

        Returns:
            (torch.Tensor):

        Examples:
            >>> nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
            >>> nested_tensor.mask
            tensor([[ True,  True,  True],
                    [ True,  True, False]])
        """

        return self._mask(tuple(self._storage))

    @classmethod
    def from_tensor_mask(cls, tensor: Tensor, mask: Tensor):
        r"""
        Build a `NestedTensor` object from a padded `Tensor` and corresponding mask `Tensor`.

        Args:
            tensor: Padded Tensor.
            mask: Tensor Mask.

        Returns:
            (torch.Tensor):

        Examples:
            >>> padded_tensor = torch.tensor([[1, 2, 3, 0, 0],
            ...                                [4, 5, 0, 0, 0],
            ...                                [6, 7, 8, 9, 0]])
            >>> mask_tensor = torch.tensor([[1, 1, 1, 0, 0],
            ...                             [1, 1, 0, 0, 0],
            ...                             [1, 1, 1, 1, 0]])
            >>> nested_tensor = NestedTensor.from_tensor_mask(padded_tensor, mask_tensor)
            >>> nested_tensor
            NestedTensor([[1, 2, 3, 0],
                    [4, 5, 0, 0],
                    [6, 7, 8, 9]])
        """

        if mask.ndim == 2:
            return cls(t[slice(0, m.sum())] for t, m in zip(tensor, mask))
        return cls(
            t[[slice(0, (m.sum(dim=dim) > 0).sum().item()) for dim in reversed(range(m.dim()))]]
            for t, m in zip(tensor, mask)
        )

    def nested_like(self, other: Tensor, unsafe: bool = False) -> NestedTensor:
        r"""
        Create a new `NestedTensor` from a `Tensor`.
        The newly created `NestedTensor` will have the same shape as current `NestedTensor`.

        Args:
            other: The `Tensor` to be nested.
            unsafe: Whether to check the shape of `other` and current `NestedTensor`.

        Returns:
            (NestedTensor):

        Examples:
            >>> nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
            >>> (nested_tensor == nested_tensor.nested_like(nested_tensor)).all()
            tensor(True)
            >>> tensor = nested_tensor.tensor
            >>> (nested_tensor == nested_tensor.nested_like(tensor)).all()
            tensor(True)
            >>> f = nested_tensor.nested_like(torch.randn(2, 2))
            Traceback (most recent call last):
            ValueError: The shape of NestedTensor and input tensor does not match, torch.Size([2, 3]) != torch.Size([2, 2])
            >>> p = nested_tensor.nested_like(torch.randn(2, 2), True)
            >>> p = nested_tensor.nested_like(torch.randn(3, 3), True)
            Traceback (most recent call last):
            ValueError: The batch size of NestedTensor and input tensor does not match, 2 != 3
        """  # noqa: E501

        if isinstance(other, NestedTensor):
            return other.clone()

        if not unsafe and self.shape != other.shape:
            raise ValueError(
                f"The shape of NestedTensor and input tensor does not match, {self.shape} != {other.shape}"
            )
        if self.size(0) != other.size(0):
            raise ValueError(
                f"The batch size of NestedTensor and input tensor does not match, {self.size(0)} != {other.size(0)}"
            )
        return NestedTensor([o[tuple(slice(0, dim) for dim in t.shape)] for t, o in zip(self._storage, other)])

    @property
    def device(self) -> torch.device:
        r"""
        Device of the NestedTensor.

        Returns:
            (torch.Tensor):

        Examples:
            >>> nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
            >>> nested_tensor.device
            device(type='cpu')
        """

        return self._device(tuple(self._storage))

    @property
    def shape(self) -> torch.Size | int:
        r"""
        Alias for `size()`.
        """

        return self.size()

    @property
    def ndim(self) -> int:
        r"""
        Alias for `dim()`.
        """

        return self.dim()

    def size(self, dim: int | None = None) -> torch.Size | int:
        r"""
        Returns the size of the self `NestedTensor`.

        Args:
            dim: If not specified, the returned value is a `torch.Size`, a subclass of `tuple`.
                If specified, returns an `int` holding the size of that dimension.
                Defaults to `None`.

        Returns:
            (torch.Size | int):

        Examples:
            >>> nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
            >>> nested_tensor.size()
            torch.Size([2, 3])
            >>> nested_tensor.size(0)
            2
            >>> nested_tensor.storage()[1] = torch.tensor([4, 5, 6, 7])
            >>> nested_tensor.shape
            torch.Size([2, 4])
            >>> nested_tensor.size(1)
            4
        """

        return self._size(tuple(self._storage), dim)

    def dim(self) -> int:
        r"""
        Number of dimension of the NestedTensor.

        Returns:
            (int):

        Examples:
            >>> nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
            >>> nested_tensor.dim()
            2
            >>> nested_tensor.storage().append(torch.tensor([6, 7, 8, 9]))
            >>> nested_tensor.ndim
            2
        """

        return self._dim(tuple(self._storage))

    def tolist(self) -> list:
        r"""
        Convert a NestedTensor to a list of lists of values.

        Returns:
            (list):

        Examples:
            >>> nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
            >>> nested_tensor.tolist()
            [[1, 2, 3], [4, 5]]
        """

        return [t.tolist() for t in self._storage]

    def all(self, dim: int | None = None, keepdim: bool = False) -> bool | Tensor | NestedTensor:
        r"""
        Tests if all elements in NestedTensor evaluate to True.

        Returns:
            (bool | Tensor):

        Examples:
            >>> nested_tensor = NestedTensor([torch.ones(2, 4, dtype=torch.bool), torch.ones(3, 5, dtype=torch.bool)])
            >>> nested_tensor.all()
            tensor(True)
            >>> nested_tensor.all(dim=0)
            tensor([True, True])
            >>> nested_tensor.all(dim=0, keepdim=True)
            tensor([[True, True]])
            >>> nested_tensor.all(dim=1)
            NestedTensor([[ True,  True,  True,  True, False],
                    [ True,  True,  True,  True,  True]])
            >>> nested_tensor.all(dim=1, keepdim=True)
            NestedTensor([[[ True,  True,  True,  True, False]],
            <BLANKLINE>
                    [[ True,  True,  True,  True,  True]]])
            >>> nested_tensor.batch_first = False
            >>> nested_tensor.all(dim=1)
            tensor([True, True])
            >>> nested_tensor.batch_first = False
            >>> nested_tensor.all(dim=0)
            NestedTensor([[ True,  True,  True,  True, False],
                    [ True,  True,  True,  True,  True]])
            >>> nested_tensor.all(dim=1)
            tensor([True, True])
        """

        if dim is None:
            return torch.tensor(all(i.all() for i in self._storage))
        if (self.batch_first and dim == 0) or (not self.batch_first and dim == 1):
            if keepdim:
                return torch.tensor([i.all() for i in self._storage]).unsqueeze(0 if self.batch_first else 1)
            return torch.tensor([i.all() for i in self._storage])
        if self.batch_first or dim != 0:
            dim -= 1
        return NestedTensor([i.all(dim=dim, keepdim=keepdim) for i in self._storage])

    def where(self, condition, other) -> NestedTensor:
        r"""
        Return a NestedTensor of elements selected from either self or other, depending on condition.

        Returns:
            (NestedTensor):

        Examples:
            >>> nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
            >>> nested_tensor.where(nested_tensor > 2, torch.tensor([[6, 5, 4], [3, 2, 1]]))
            NestedTensor([[6, 5, 3],
                    [4, 5, 0]])
            >>> nested_tensor.where(nested_tensor > 2, NestedTensor([[6, 5, 4], [3, 2]]))
            NestedTensor([[6, 5, 3],
                    [4, 5, 0]])
            >>> nested_tensor.where(nested_tensor > 2, torch.nan)
            NestedTensor([[nan, nan, 3.],
                    [4., 5., 0.]])
            >>> nested_tensor.where(nested_tensor.tensor > 2, torch.nan)
            NestedTensor([[nan, nan, 3.],
                    [4., 5., 0.]])
            >>> nested_tensor.where(torch.tensor(True), NestedTensor([[6, 5, 4], [3, 2]]))
            NestedTensor([[1, 2, 3],
                    [4, 5, 0]])
            >>> nested_tensor.where(torch.tensor(False), 1)
            NestedTensor([[1, 1, 1],
                    [1, 1, 0]])
        """

        if isinstance(condition, Tensor) and self.shape == condition.shape:
            condition = self.nested_like(condition)
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(condition, NestedTensor) and isinstance(other, NestedTensor):
            return NestedTensor(
                [x.where(c, y) for x, c, y in zip(self._storage, condition._storage, other._storage)], **self._state
            )
        if isinstance(condition, NestedTensor):
            return NestedTensor([x.where(c, other) for x, c in zip(self._storage, condition._storage)], **self._state)
        if isinstance(other, NestedTensor):
            return NestedTensor([x.where(condition, y) for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor(x.where(condition, other) for x in self._storage)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in NestedTensorFunc or not all(issubclass(t, (torch.Tensor, NestedTensor)) for t in types):
            args = [a.tensor if hasattr(a, "tensor") else a for a in args]
            return func(*args, **kwargs)
        return NestedTensorFunc[func](*args, **kwargs)

    def __getitem__(self, index: int | slice | tuple) -> tuple[Tensor, Tensor] | NestedTensor:
        if isinstance(index, tuple):
            return NestedTensor([t[index[0]][index[1:]] for t in self._storage])
        if isinstance(index, (int, slice)):
            ret = self._storage[index]
            if isinstance(ret, Tensor):
                return ret, torch.ones_like(ret, dtype=torch.bool)
            return self.tensor, self.mask
        raise ValueError(f"Unsupported index type {type(index)}")

    def __getattr__(self, name) -> Any:
        if not self._storage:
            raise ValueError(f"Unable to get {name} from an empty {self.__class__.__name__}")
        ret = [getattr(i, name) for i in self._storage]
        elem = ret[0]
        if isinstance(elem, Tensor):
            return NestedTensor(ret, **self._state)
        if callable(elem):
            return NestedTensorFuncWrapper(ret, state=self._state)
        if elem.__hash__ is not None and len(set(ret)) == 1:
            return elem
        return ret

    @property
    def _state(self) -> Mapping:
        return {k: v for k, v in self.__dict__.items() if not (k.startswith("_") or k.endswith("_"))}

    def __state__(self) -> Mapping:
        return self.__dict__

    def __setstate__(self, state: Mapping) -> None:
        self.__dict__.update(state)

    def __len__(self) -> int:
        return len(self._storage)

    def __repr__(self):
        return self.__class__.__name__ + repr(self.tensor)[len(self.tensor.__class__.__name__) :]  # noqa: E203

    def __bool__(self) -> int:
        return all(bool(x) for x in self._storage)

    def __gt__(  # type: ignore[override]
        self, other: Tensor | NestedTensor | SupportsFloat
    ) -> bool | Tensor | NestedTensor:
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor(i > j for i, j in zip(self._storage, other._storage))
        if isinstance(other, SupportsFloat):
            return NestedTensor([x > other for x in self._storage], **self._state)
        raise TypeError(f"> not supported between instances of '{type(self)}' and '{type(other)}'")

    def __ge__(  # type: ignore[override]
        self, other: Tensor | NestedTensor | SupportsFloat
    ) -> bool | Tensor | NestedTensor:
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor(i >= j for i, j in zip(self._storage, other._storage))
        if isinstance(other, SupportsFloat):
            return NestedTensor([x >= other for x in self._storage], **self._state)
        raise TypeError(f">= not supported between instances of '{type(self)}' and '{type(other)}'")

    def __eq__(  # type: ignore[override]
        self, other: Tensor | NestedTensor | SupportsFloat
    ) -> bool | Tensor | NestedTensor:
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor(i == j for i, j in zip(self._storage, other._storage))
        if isinstance(other, SupportsFloat):
            return NestedTensor([x == other for x in self._storage], **self._state)
        return False

    def __le__(  # type: ignore[override]
        self, other: Tensor | NestedTensor | SupportsFloat
    ) -> bool | Tensor | NestedTensor:
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor(i <= j for i, j in zip(self._storage, other._storage))
        if isinstance(other, SupportsFloat):
            return NestedTensor([x <= other for x in self._storage], **self._state)
        raise TypeError(f"<= not supported between instances of '{type(self)}' and '{type(other)}'")

    def __lt__(  # type: ignore[override]
        self, other: Tensor | NestedTensor | SupportsFloat
    ) -> bool | Tensor | NestedTensor:
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor(i < j for i, j in zip(self._storage, other._storage))
        if isinstance(other, SupportsFloat):
            return NestedTensor([x < other for x in self._storage], **self._state)
        raise TypeError(f"<= not supported between instances of '{type(self)}' and '{type(other)}'")

    def __abs__(self):
        return NestedTensor([abs(value) for value in self._storage], **self._state)

    def __add__(self, other):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([x + y for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([value + other for value in self._storage], **self._state)

    def __radd__(self, other):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([y + x for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([other + value for value in self._storage], **self._state)

    def __iadd__(self, other):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if hasattr(other, "to"):
            other = other.to(self.dtype)
        if isinstance(other, NestedTensor):
            for x, y in zip(self._storage, other._storage):
                x += y
        else:
            for value in self._storage:
                value += other
        return self

    def __and__(self, other):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([x & y for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([value & other for value in self._storage], **self._state)

    def __rand__(self, other):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([y & x for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([other & value for value in self._storage], **self._state)

    def __iand__(self, other):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if hasattr(other, "to"):
            other = other.to(self.dtype)
        if isinstance(other, NestedTensor):
            for x, y in zip(self._storage, other._storage):
                x &= y
        else:
            for value in self._storage:
                value &= other
        return self

    def __floordiv__(self, other):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([x // y for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([value // other for value in self._storage], **self._state)

    def __rfloordiv__(self, other):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([y // x for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([other // value for value in self._storage], **self._state)

    def __ifloordiv__(self, other):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if hasattr(other, "to"):
            other = other.to(self.dtype)
        if isinstance(other, NestedTensor):
            for x, y in zip(self._storage, other._storage):
                x //= y
        else:
            for value in self._storage:
                value //= other
        return self

    def __mod__(self, other):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([x % y for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([value % other for value in self._storage], **self._state)

    def __rmod__(self, other):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([y % x for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([other % value for value in self._storage], **self._state)

    def __imod__(self, other):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if hasattr(other, "to"):
            other = other.to(self.dtype)
        if isinstance(other, NestedTensor):
            for x, y in zip(self._storage, other._storage):
                x %= y
        else:
            for value in self._storage:
                value %= other
        return self

    def __mul__(self, other):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([x * y for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([value * other for value in self._storage], **self._state)

    def __rmul__(self, other):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([y * x for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([other * value for value in self._storage], **self._state)

    def __imul__(self, other):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if hasattr(other, "to"):
            other = other.to(self.dtype)
        if isinstance(other, NestedTensor):
            for x, y in zip(self._storage, other._storage):
                x *= y
        else:
            for value in self._storage:
                value *= other
        return self

    def __matmul__(self, other):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([x @ y for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([value @ other for value in self._storage], **self._state)

    def __rmatmul__(self, other):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([y @ x for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([other @ value for value in self._storage], **self._state)

    def __imatmul__(self, other):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if hasattr(other, "to"):
            other = other.to(self.dtype)
        if isinstance(other, NestedTensor):
            for x, y in zip(self._storage, other._storage):
                x @= y
        else:
            for value in self._storage:
                value @= other
        return self

    def __pow__(self, other):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([x**y for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([value**other for value in self._storage], **self._state)

    def __rpow__(self, other):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([y**x for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([other**value for value in self._storage], **self._state)

    def __ipow__(self, other):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if hasattr(other, "to"):
            other = other.to(self.dtype)
        if isinstance(other, NestedTensor):
            for x, y in zip(self._storage, other._storage):
                x *= y
        else:
            for value in self._storage:
                value *= other
        return self

    def __truediv__(self, other):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([x / y for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([value / other for value in self._storage], **self._state)

    def __rtruediv__(self, other):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([y / x for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([other / value for value in self._storage], **self._state)

    def __itruediv__(self, other):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if hasattr(other, "to"):
            other = other.to(self.dtype)
        if isinstance(other, NestedTensor):
            for x, y in zip(self._storage, other._storage):
                x /= y
        else:
            for value in self._storage:
                value /= other
        return self

    def __sub__(self, other):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([x - y for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([value - other for value in self._storage], **self._state)

    def __rsub__(self, other):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([y - x for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([other - value for value in self._storage], **self._state)

    def __isub__(self, other):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if hasattr(other, "to"):
            other = other.to(self.dtype)
        if isinstance(other, NestedTensor):
            for x, y in zip(self._storage, other._storage):
                x -= y
        else:
            for value in self._storage:
                value -= other
        return self

    @method_cache(maxsize=1)
    def _tensor(self, storage) -> Tensor:
        if storage[0].dim() == 0:
            return torch.stack(storage, dim=0)
        return pad_tensor(
            storage, size=self.size(), batch_first=self.batch_first, padding_value=float(self.padding_value)
        )

    @method_cache(maxsize=1)
    def _mask(self, storage) -> Tensor:
        if storage[0].dim() == 0:
            return torch.full(len(storage), fill_value=not self.mask_value, dtype=torch.bool, device=storage[0].device)
        size = self.size()
        # ignore channel dimension
        if storage[0].dim() > 1 and len({t.size(-1) for t in storage}) == 1:
            size = size[:-1]  # type: ignore
        return mask_tensor(storage, size=size, batch_first=self.batch_first, mask_value=self.mask_value)

    @method_cache(maxsize=1)
    def _device(self, storage) -> torch.device:
        return storage[0].device

    @method_cache(maxsize=1)
    def _size(self, storage, dim: int | None = None) -> torch.Size | int:
        if dim is not None:
            if dim == 0:
                return len(storage)
            return max(t.size(dim - 1) for t in storage)
        if max(t.dim() for t in storage) == 0:
            return torch.Size((len(storage),))
        ndim = max(t.dim() for t in storage)
        size = [max(t.shape[i] if i < len(t.shape) else 0 for t in storage) for i in range(ndim)]
        size.insert(0 if self.batch_first else 1, len(storage))
        return torch.Size(size)

    @method_cache(maxsize=1)
    def _dim(self, storage) -> torch.Size:
        return max(t.dim() for t in storage) + 1


NestedTensorFunc = TorchFuncRegistry()


@NestedTensorFunc.implement(torch.mean)
def mean(
    input,
    dim: int | None = None,
    keepdim: bool = False,
    *,
    dtype: torch.dtype | None = None,
):
    return input.mean(dim=dim, keepdim=keepdim, dtype=dtype)


@NestedTensorFunc.implement(torch.cat)
def cat(tensors, dim: int = 0):
    if dim != 0:
        raise NotImplementedError(f"NestedTensor only supports cat when dim=0, but got {dim}")
    return NestedTensor([t for tensor in tensors for t in tensor._storage], tensors[0]._state)


@NestedTensorFunc.implement(torch.stack)
def stack(tensors, dim: int = 0):
    raise NotImplementedError("NestedTensor does not support stack as of now")


@NestedTensorFunc.implement(torch.isin)
def isin(elements, test_elements, *, assume_unique: bool = False, invert: bool = False):
    if isinstance(elements, NestedTensor):
        elements = elements.tensor
    if isinstance(test_elements, NestedTensor):
        test_elements = test_elements.tensor
    return torch.isin(elements, test_elements, assume_unique=assume_unique, invert=invert)


class NestedTensorFuncWrapper:
    r"""
    Function Wrapper to handle NestedTensor as input.
    """

    _storage: Sequence[Callable] = []
    state: Mapping = {}

    def __init__(self, callables, state: Mapping | None = None) -> None:
        if not isinstance(callables, Sequence):
            raise ValueError(f"NestedTensorFuncWrapper must be initialised with a Sequence, bug got {type(callables)}")
        if len(callables) == 0:
            raise ValueError("NestedTensorFuncWrapper must be initialised with a non-empty Sequence.")
        if not callable(callables[0]):
            raise ValueError(
                f"NestedTensorFuncWrapper must be initialised with a Sequence of Callable, bug got {type(callables[0])}"
            )
        self._storage = callables
        if state is None:
            state = {}
        self.state = state

    def __call__(self, *args, **kwargs) -> NestedTensor | Sequence[Tensor]:
        ret = [call(*args, **kwargs) for call in self._storage]
        elem = ret[0]
        if isinstance(elem, Tensor):
            return NestedTensor(ret, **self.state)
        if elem.__hash__ is not None and len(set(ret)) == 1:
            return elem
        return ret


def collate_pn_tensor_fn(batch, *, collate_fn_map: dict[type | tuple[type, ...], Callable] | None = None):
    return NestedTensor(batch)


default_collate_fn_map[PNTensor] = collate_pn_tensor_fn
