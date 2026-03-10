"""Run the API server: python -m eval_lab.api"""

import uvicorn

from eval_lab.api.app import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
