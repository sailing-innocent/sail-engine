import aiohttp 
from aiohttp import web 
from argparse import ArgumentParser
import asyncio
import os 
from loguru import logger 

class WebUIServer():
    def __init__(self, loop):
        self.web_root = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), "web")
        routes = web.RouteTableDef()
        self.routes = routes
        self.app = web.Application()
        self.loop = loop 
        self.messages = asyncio.Queue()

        @routes.get('/')
        async def index(request):
            return web.FileResponse(os.path.join(self.web_root, "index.html"))

    def add_routes(self):
        self.app.add_routes(self.routes)
        self.app.add_routes([
            web.static('/', self.web_root),
        ])
    
    async def send(self, event, data, sid=None):
        pass 

    async def publish_loop(self):
        while True:
            msg = await self.messages.get()
            await self.send(*msg)

    async def start(self, address, port, verbose=True, call_on_start=None):
        runner = web.AppRunner(self.app, access_log=None)
        await runner.setup()
        ssl_ctx = None
        scheme = "http"
        site = web.TCPSite(runner, address, port, ssl_context=ssl_ctx)
        await site.start()
        if verbose:
            logger.info("Starting server\n")
            logger.info("To see the GUI go to: {}://{}:{}".format(scheme, address, port))
        if call_on_start is not None:
            call_on_start(scheme, address, port)

async def run(server, address='', port=8188, verbose=True, call_on_start=None):
    await asyncio.gather(server.start(address, port, verbose, call_on_start), server.publish_loop())
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--address", type=str, default="127.0.0.1")
    args = parser.parse_args()


    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server = WebUIServer(loop)
    server.add_routes()
    call_on_start = None 
    try:
        loop.run_until_complete(run(server, address=args.address, port=args.port, call_on_start=call_on_start))
    except KeyboardInterrupt:
        logger.info("Shutting down server")

    
    