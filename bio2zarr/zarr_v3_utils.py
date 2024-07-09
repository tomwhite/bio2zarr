from dataclasses import dataclass

import numcodecs
from numcodecs.compat import ensure_bytes, ensure_ndarray
from zarr.abc.codec import ArrayBytesCodec
from zarr.array_spec import ArraySpec
from zarr.buffer import Buffer, NDBuffer
from zarr.codecs.registry import register_codec
from zarr.common import JSON, to_thread


@dataclass(frozen=True)
class VLenUTF8Codec(ArrayBytesCodec):
    is_fixed_size = False

    def __init__(self, *args, **kwargs) -> None:
        pass

    def to_dict(self) -> dict[str, JSON]:
        return {"name": "vlen-utf8", "compressor": {"id": "vlen-utf8"}}

    async def _decode_single(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> NDBuffer:
        compressor = numcodecs.get_codec(dict(id="vlen-utf8"))
        chunk_numpy_array = ensure_ndarray(
            await to_thread(compressor.decode, chunk_bytes.as_array_like())
        )

        # ensure correct dtype
        if str(chunk_numpy_array.dtype) != chunk_spec.dtype:
            chunk_numpy_array = chunk_numpy_array.view(chunk_spec.dtype)

        # ensure correct chunk shape
        if chunk_numpy_array.shape != chunk_spec.shape:
            chunk_numpy_array = chunk_numpy_array.reshape(
                chunk_spec.shape,
            )

        return NDBuffer.from_numpy_array(chunk_numpy_array)

    async def _encode_single(
        self,
        chunk_array: NDBuffer,
        _chunk_spec: ArraySpec,
    ) -> Buffer | None:
        chunk_numpy_array = chunk_array.as_numpy_array()
        compressor = numcodecs.get_codec(dict(id="vlen-utf8"))
        if (
            not chunk_numpy_array.flags.c_contiguous
            and not chunk_numpy_array.flags.f_contiguous
        ):
            chunk_numpy_array = chunk_numpy_array.copy(order="A")
        encoded_chunk_bytes = ensure_bytes(
            await to_thread(compressor.encode, chunk_numpy_array)
        )

        return Buffer.from_bytes(encoded_chunk_bytes)

    def compute_encoded_size(
        self, _input_byte_length: int, _chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError


register_codec("vlen-utf8", VLenUTF8Codec)
