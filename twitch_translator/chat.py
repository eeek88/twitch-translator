"""Anonymous read-only Twitch chat client over raw IRC.

Twitch allows anonymous reads with a "justinfan" nick — no OAuth or API key.
"""
from __future__ import annotations

import socket
from typing import Iterator, Optional

HOST = "irc.chat.twitch.tv"
PORT = 6667


class ChatReader:
    """One IRC connection to one channel. stop() can be called from another
    thread to interrupt a blocked read immediately — needed for switching
    channels live instead of waiting out the old connection's timeout.

    shutdown() before close() matters specifically on Windows: closing a
    socket that another thread is blocked inside recv() on doesn't reliably
    unblock it the way it does on POSIX. shutdown() does.
    """

    def __init__(self, channel: str):
        self.channel = channel.lower().lstrip("#")
        self._sock: Optional[socket.socket] = None

    def stop(self):
        sock = self._sock
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()

    def messages(self) -> Iterator[tuple[str, str]]:
        """Yields (username, message) for each chat message in the channel.
        Blocks until stop()'d or the connection drops (caller handles retry)."""
        sock = socket.create_connection((HOST, PORT), timeout=30)
        self._sock = sock
        try:
            # Twitch PINGs roughly every 5 minutes even in a silent chat, so total
            # silence for 6+ minutes means the connection is dead. (The 30s connect
            # timeout above would otherwise apply to every recv and trip on quiet chats.)
            sock.settimeout(360)
            sock.sendall(b"NICK justinfan12345\r\n")
            sock.sendall(f"JOIN #{self.channel}\r\n".encode())

            buf = b""
            while True:
                data = sock.recv(4096)
                if not data:
                    raise ConnectionError("Twitch IRC connection closed")
                buf += data
                while b"\r\n" in buf:
                    line, buf = buf.split(b"\r\n", 1)
                    text = line.decode("utf-8", errors="replace")
                    if text.startswith("PING"):
                        sock.sendall(text.replace("PING", "PONG", 1).encode() + b"\r\n")
                        continue
                    # :nick!nick@nick.tmi.twitch.tv PRIVMSG #channel :message text
                    if " PRIVMSG " not in text:
                        continue
                    prefix, _, rest = text.partition(" PRIVMSG ")
                    username = prefix.lstrip(":").split("!", 1)[0]
                    _, _, message = rest.partition(" :")
                    if message:
                        yield username, message
        finally:
            sock.close()


def chat_messages(channel: str) -> Iterator[tuple[str, str]]:
    """Convenience wrapper for one-shot use (e.g. scripts/test_chat.py)."""
    return ChatReader(channel).messages()
