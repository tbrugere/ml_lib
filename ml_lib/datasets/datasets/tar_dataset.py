from typing import TypeVar, IO, ClassVar
import io
from os import PathLike
from pathlib import Path
import tarfile
import yaml
import zipfile
from torch.utils.data import get_worker_info

from ..base_classes import Dataset
from ..datapoint import Datapoint, DictDatapoint
from ..registration import register



PointType = TypeVar("PointType", bound=Datapoint)

class DataReadingError(Exception):
    pass

class TarDataset(Dataset[PointType]):
    """
    Reads the data from a tar archive file 
    (which is a nice way of storing your data in a "multiple files" style
     without actually needing file descriptors for every point,
     which can be kinda slow, or overloading your inodes, etc.)

    Q/A

    In what order does it retrieve the data?
        In the default order of tarfile.getmembers(). I think alphabetical order.

    Does it read all the data to RAM?
        No. But you can pipe it to CacheTransform if that's what you wish.

    Can I use several workers with this?:
        Currently no there's a bug I still need to fix

    What isn't safe?
        please don't edit the tar file while reading from it?
    """
    metadata_file: ClassVar[str] = "__metadata__"

    file_path: PathLike

    members_list: list[str]
    
    # at first I thought it would be enough to not have it in getstate (so it wouldn't be sent to workers)
    # but i think some workers are created by forking
    _file: dict[None|int, tarfile.TarFile] 

    # Tarfiles do not have an index, 
    # so if we do the naive thing, we end up going through the whole file every time
    # we need to pull up a datapoint
    # instead we keep an index of TarInfo objects (which include the offset)
    _index: dict[str, tarfile.TarInfo]

    metadata: dict

    def __init__(self, tar_file: PathLike):
        self.file_path = tar_file
        self._file = {}
        file = self.tar_file # actually initializes the _file
        members_list = file.getnames()
        self._index = {
                member.name: member for member in file.getmembers()
            }
        if self.metadata_file in members_list:
            members_list = [member for member in members_list 
                            if member != self.metadata_file]
            metadata_file = file.extractfile(self.metadata_file)
            assert metadata_file is not None
            self.metadata = yaml.safe_load(metadata_file)
        else: 
            self.metadata = dict()
        self.members_list = members_list

    def __getitem__(self, i):
        member_name = self.members_list[i]
        member = self._index[member_name]
        file = self.tar_file
        reader = file.extractfile(member)
        if reader is None:
            raise ValueError(f"member {member_name} in file {self.file_path} is not a file or a link. This is unsupported.")
        try:
            reader.seek(0)
            return self.read_element(reader)
        except zipfile.BadZipFile as e:
            reader.seek(0)
            data = reader.read()
            Path("/tmp/file_error").write_bytes(data)
            raise DataReadingError(f"while trying to read {file}, content written to /tmp/file_error") from e

    def __len__(self):
        return len(self.members_list)

    def read_element(self, file: IO[bytes]) -> PointType:
        del file
        raise NotImplementedError(f"you cannot use {self.__class__} directly, you need to overload  the read_element function that reads data from one of the files in the tarfile")

    @property
    def tar_file(self) -> tarfile.TarFile:
        workerid = self._get_workerid()
        if workerid in self._file:
            return self._file[workerid]
        file = tarfile.open(self.file_path)
        self._file[workerid] = file
        return file

    @staticmethod
    def read_metadata(metadata_file):
        """Can be edited if your metadata file is not in yaml format"""
        return yaml.safe_load(metadata_file)

    def _get_workerid(self):
        worker_info = get_worker_info()
        if worker_info is None: return None
        return worker_info.id

    def __getstate__(self):
        return dict(file_path=self.file_path, 
                    members_list = self.members_list, 
                    index = self._index)

    def __setstate__(self, d):
        self.file_path = d["file_path"]
        self.members_list = d["members_list"]
        self._file = {}
        self._index = d["index"]
        
    def __repr__(self):
        return f"{self.__class__.__name__}({str(self.file_path)})"

    def dataset_parameters(self):
        return {**self.metadata}

        
@register
class TarNpzDataset(TarDataset[DictDatapoint]):
    """
    Dataset that takes a tar file containing npz files 
    (compressed numpy tensors).
    The numpy vectors are transformed to torch tensors, and returned
    as DictDatapoints
    """
    datatype = DictDatapoint

    def read_element(self, file: IO[bytes]):
        import numpy as np
        with np.load(file) as npz:
            result = {f: npz[f] for f in npz.files}
        return DictDatapoint(result)

def add_file_to_tarfile(file, bytes_io, tar_filename):
    tar_fileinfo = tarfile.TarInfo(name=tar_filename)
    tar_fileinfo.size = bytes_io.getbuffer().nbytes
    bytes_io.seek(0)
    file.addfile(tar_fileinfo, bytes_io)

@register
class AutoTarDataset(TarDataset):

    def __init__(self, datatype: type[Datapoint], tar_file: PathLike):
        self.datatype = datatype
        super().__init__(tar_file)

    def read_element(self, file: IO[bytes]):
        import numpy as np
        import torch
        with np.load(file) as npz:
            result = {f: torch.as_tensor(npz[f]) 
                      for f in npz.files}
        return self.datatype(**result)

    @classmethod
    def save_dataset(cls, tar_file: PathLike, dataset: Dataset):
        import numpy as np
        from math import log10, ceil

        n_digits = int(ceil(log10(len(dataset))))
        try: 
            with tarfile.open(tar_file, "w") as tar:
                for i, point in enumerate(dataset):
                    tar_filename = f"{i:0{n_digits}d}.npz"
                    bytes_io = io.BytesIO()
                    np.savez(bytes_io, **point.asdict())
                    add_file_to_tarfile(tar, bytes_io, tar_filename)
                metadata = dataset.dataset_parameters()
                metadata_file = io.BytesIO(yaml.dump(metadata).encode())
                add_file_to_tarfile(tar, metadata_file, cls.metadata_file)
        except:
            # if something goes wrong
            # we don't want to leave a half-written file
            Path(tar_file).unlink()
            raise



class TarTorchDataset(TarDataset[DictDatapoint]):
    """
    Dataset that takes a tar file containing pt files 
    (pickle files saved with torch.save).
    The numpy vectors are transformed to torch tensors, and returned
    as DictDatapoints
    """
    datatype = DictDatapoint

    def read_element(self, file: IO[bytes]):
        import torch
        result = torch.load(file)
        if not isinstance(result, dict):
            result = {"x": result}

        return DictDatapoint(result)
