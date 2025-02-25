from src.data_loader import allen_api
from dotenv import load_dotenv

load_dotenv()

boc = allen_api.get_boc()

print(boc)
