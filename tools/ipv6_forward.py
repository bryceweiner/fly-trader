"""Forward [::1]:PORT to 127.0.0.1:PORT for each port given (the local Solana validator listens on IPv4 only, but
browsers and Phantom's "Solana Localnet" often resolve localhost to ::1 first).

    python tools/ipv6_forward.py 8899 8900
"""
import asyncio
import sys


async def pipe(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
    try:
        while data := await r.read(65536):
            w.write(data)
            await w.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        w.close()


def handler(port: int):
    async def handle(cr: asyncio.StreamReader, cw: asyncio.StreamWriter) -> None:
        try:
            ur, uw = await asyncio.open_connection("127.0.0.1", port)
        except OSError:
            cw.close()
            return
        await asyncio.gather(pipe(cr, uw), pipe(ur, cw))
    return handle


async def main(ports: list[int]) -> None:
    servers = [await asyncio.start_server(handler(p), "::1", p) for p in ports]
    await asyncio.gather(*(s.serve_forever() for s in servers))


if __name__ == "__main__":
    asyncio.run(main([int(p) for p in sys.argv[1:]]))
