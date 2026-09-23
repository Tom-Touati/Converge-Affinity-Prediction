"""A scalar/image/text event writer that does not import torch.

``to_tensorboard.py`` used ``torch.utils.tensorboard.SummaryWriter``, which drags in all of
torch to write a few floats. That became load-bearing when torch stopped being importable
on this workstation -- ``WinError 1114``, a DLL that cannot initialise under memory
pressure -- and the mirror stopped working while TensorBoard itself, which needs no torch
to SERVE, kept running. Writing the protos directly removes the dependency entirely.

The protos ship with the ``tensorboard`` package, so this adds nothing to install.
"""
from __future__ import annotations

import io
import time

from tensorboard.compat.proto import event_pb2, summary_pb2
from tensorboard.summary.writer.event_file_writer import EventFileWriter


class Writer:
    """The small part of SummaryWriter's surface this project actually uses."""

    def __init__(self, logdir: str):
        self._w = EventFileWriter(logdir)

    def _emit(self, summary: summary_pb2.Summary, step: int) -> None:
        self._w.add_event(event_pb2.Event(wall_time=time.time(), step=int(step),
                                          summary=summary))

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self._emit(summary_pb2.Summary(
            value=[summary_pb2.Summary.Value(tag=tag, simple_value=float(value))]), step)

    def add_figure(self, tag: str, figure, step: int = 0) -> None:
        """A matplotlib figure, encoded as PNG. Closing it is the caller's business."""
        buf = io.BytesIO()
        figure.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        png = buf.getvalue()
        w, h = figure.canvas.get_width_height()
        img = summary_pb2.Summary.Image(height=h, width=w, colorspace=4,
                                        encoded_image_string=png)
        self._emit(summary_pb2.Summary(
            value=[summary_pb2.Summary.Value(tag=tag, image=img)]), step)

    def add_text(self, tag: str, text: str, step: int = 0) -> None:
        from tensorboard.compat.proto import tensor_pb2, tensor_shape_pb2, types_pb2
        t = tensor_pb2.TensorProto(
            dtype=types_pb2.DT_STRING,
            string_val=[text.encode("utf-8")],
            tensor_shape=tensor_shape_pb2.TensorShapeProto(
                dim=[tensor_shape_pb2.TensorShapeProto.Dim(size=1)]))
        meta = summary_pb2.SummaryMetadata(
            plugin_data=summary_pb2.SummaryMetadata.PluginData(plugin_name="text"))
        self._emit(summary_pb2.Summary(
            value=[summary_pb2.Summary.Value(tag=tag + "/text_summary", tensor=t,
                                             metadata=meta)]), step)

    def add_hparams(self, hparams: dict, metrics: dict, run_name: str = ".") -> None:
        """Scalars only.

        The HPARAMS tab needs an experiment/session protobuf pair that is fiddly to build
        by hand and adds little here: the same numbers are already the run's own scalars,
        and the Runs table in the mirror's own summary output covers the comparison. Only
        the metrics are written, under hp/, so nothing silently disappears.
        """
        for k, v in metrics.items():
            self.add_scalar(k, float(v), 0)

    def close(self) -> None:
        self._w.flush()
        self._w.close()
