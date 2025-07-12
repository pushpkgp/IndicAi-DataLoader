# start.py
import yaml
import uvicorn

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    cfg = load_config()
    host = cfg['app'].get('host', '127.0.0.1')
    port = cfg['app'].get('port', 8000)

    uvicorn.run("app.main:app", host=host, port=port)