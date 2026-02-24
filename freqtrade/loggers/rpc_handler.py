import logging

from freqtrade.enums import RPCMessageType


class RPCHandler(logging.Handler):
    """
    Logging handler that forwards messages to the RPC manager.
    """

    def __init__(self, rpc_manager):
        super().__init__()
        self.rpc_manager = rpc_manager
        self._is_sending = False

    def emit(self, record):
        if (
            self._is_sending
            or record.name.startswith("freqtrade.rpc")
            or not self.rpc_manager._rpc._freqtrade
        ):
            return

        # Skip noisy network errors to avoid timeout loops
        msg = self.format(record)
        skip_words = ["Timed out", "RequestTimeout", "status_code 520", "RemoteDisconnected"]
        if any(word in msg for word in skip_words):
            return

        # Don't try to send to RPC if bot is in a state that doesn't allow it
        if self.rpc_manager._rpc._freqtrade.state.value in ("stopping", "stopped"):
            return

        try:
            self._is_sending = True
            msg = self.format(record)
            if record.levelno >= logging.ERROR:
                msg_type = RPCMessageType.EXCEPTION
            elif record.levelno >= logging.WARNING:
                msg_type = RPCMessageType.WARNING
            else:
                msg_type = RPCMessageType.STATUS

            self.rpc_manager.send_msg({"type": msg_type, "status": msg})
        except Exception:
            self.handleError(record)
        finally:
            self._is_sending = False
