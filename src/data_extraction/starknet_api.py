from starknet_py.net.full_node_client import FullNodeClient
from starknet_py.net.models.chains import StarknetChainId
from typing import Dict, List, Optional
import asyncio
import logging
from datetime import datetime

from src.config import STARKNET_NODE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StarknetDataExtractor:
    def __init__(self):
        """Initialize Starknet client."""
        try:
            self.client = GatewayClient(STARKNET_NODE)
            self.chain_id = StarknetChainId.MAINNET
            logger.info("Connected to Starknet gateway")
        except Exception as e:
            logger.error(f"Failed to initialize Starknet client: {str(e)}")
            raise

    async def get_transaction(self, tx_hash: str) -> Optional[Dict]:
        """Fetch transaction details from Starknet."""
        try:
            tx = await self.client.get_transaction(tx_hash)
            receipt = await self.client.get_transaction_receipt(tx_hash)

            return {
                'hash': tx_hash,
                'type': tx.tx_type.name,
                'sender_address': hex(tx.sender_address),
                'entry_point_selector': hex(tx.entry_point_selector) if tx.entry_point_selector else None,
                'status': receipt.status.name,
                'actual_fee': receipt.actual_fee,
                'block_number': receipt.block_number,
                'block_hash': hex(receipt.block_hash) if receipt.block_hash else None,
                'timestamp': datetime.now().timestamp()  # Placeholder as Starknet lacks timestamp
            }
        except Exception as e:
            logger.error(f"Error fetching Starknet transaction {tx_hash}: {str(e)}")
            return None

    async def get_contract_state(self, contract_address: str) -> Optional[Dict]:
        """Fetch contract state from Starknet."""
        try:
            state = await self.client.get_contract_state(contract_address)
            return {
                'contract_address': contract_address,
                'storage_commitment': hex(state.storage_commitment),
                'nonce': state.nonce
            }
        except Exception as e:
            logger.error(f"Error fetching contract state {contract_address}: {str(e)}")
            return None

    async def monitor_blocks(self, callback):
        """Monitor new blocks on Starknet."""
        try:
            last_block = await self.client.get_block('latest')

            while True:
                current_block = await self.client.get_block('latest')
                if current_block.block_number > last_block.block_number:
                    await callback(current_block)
                    last_block = current_block
                await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"Error monitoring blocks: {str(e)}")

    async def get_events(self, contract_address: str, from_block: int) -> List[Dict]:
        """Fetch events from a Starknet contract."""
        try:
            events = await self.client.get_events(
                address=contract_address,
                from_block=from_block,
                to_block='latest'
            )

            return [{
                'block_number': event.block_number,
                'block_hash': hex(event.block_hash),
                'transaction_hash': hex(event.transaction_hash),
                'event_key': hex(event.keys[0]) if event.keys else None,
                'data': [hex(d) for d in event.data]
            } for event in events]
        except Exception as e:
            logger.error(f"Error fetching events from {contract_address}: {str(e)}")
            return []
