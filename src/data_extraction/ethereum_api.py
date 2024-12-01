from web3 import Web3
from typing import Dict, List, Optional
import os
import json
import asyncio
from datetime import datetime
import logging
from src.config import ALCHEMY_URL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EthereumDataExtractor:
    def __init__(self):
        """Initialize Ethereum client using Alchemy."""
        self.w3 = Web3(Web3.HTTPProvider(ALCHEMY_URL))
        if not self.w3.is_connected():
            logger.error("Failed to connect to Ethereum network via Alchemy")
            raise ConnectionError("Unable to connect to Ethereum network")
        logger.info("Connected to Ethereum network via Alchemy")

    async def get_transaction(self, tx_hash: str) -> Optional[Dict]:
        """Fetch transaction details by hash."""
        try:
            tx = self.w3.eth.get_transaction(tx_hash)
            receipt = self.w3.eth.get_transaction_receipt(tx_hash)
            block = self.w3.eth.get_block(tx['blockNumber'])

            return {
                'hash': tx_hash,
                'from': tx['from'],
                'to': tx['to'],
                'value': self.w3.from_wei(tx['value'], 'ether'),
                'gas_price': self.w3.from_wei(tx['gasPrice'], 'gwei'),
                'gas_used': receipt['gasUsed'],
                'block_number': tx['blockNumber'],
                'timestamp': block['timestamp'],
                'status': receipt['status']
            }
        except Exception as e:
            logger.error(f"Error fetching transaction {tx_hash}: {str(e)}")
            return None

    async def get_defi_protocol_events(self, contract_address: str, from_block: int) -> List[Dict]:
        """Fetch DeFi protocol events from a specific contract."""
        try:
            with open('data/raw/defi_abi.json', 'r') as f:
                contract_abi = json.load(f)

            contract = self.w3.eth.contract(
                address=self.w3.to_checksum_address(contract_address),
                abi=contract_abi
            )

            logs = contract.events.SomeEvent.create_filter(
                fromBlock=from_block,
                toBlock='latest'
            ).get_all_entries()

            return [{
                'event_type': log.event,
                'block_number': log.blockNumber,
                'transaction_hash': log.transactionHash.hex(),
                'args': dict(log.args),
                'timestamp': self.w3.eth.get_block(log.blockNumber)['timestamp']
            } for log in logs]
        except Exception as e:
            logger.error(f"Error fetching events from contract {contract_address}: {str(e)}")
            return []

    async def monitor_pending_transactions(self, callback):
        """Monitor pending transactions in real-time."""
        try:
            pending_filter = self.w3.eth.filter('pending')

            async def handle_pending(tx_hash):
                tx = await self.get_transaction(tx_hash)
                if tx:
                    await callback(tx)

            while True:
                for tx_hash in pending_filter.get_new_entries():
                    await handle_pending(tx_hash)
                await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Error monitoring pending transactions: {str(e)}")

    def get_gas_statistics(self, block_range: int = 100) -> Dict:
        """Calculate gas statistics from recent blocks."""
        try:
            latest_block = self.w3.eth.block_number
            gas_prices = []

            for block_number in range(latest_block - block_range, latest_block):
                block = self.w3.eth.get_block(block_number, True)
                for tx in block.transactions or []:
                    gas_prices.append(self.w3.from_wei(tx['gasPrice'], 'gwei'))

            return {
                'average': sum(gas_prices) / len(gas_prices) if gas_prices else 0,
                'median': sorted(gas_prices)[len(gas_prices) // 2] if gas_prices else 0,
                'min': min(gas_prices) if gas_prices else 0,
                'max': max(gas_prices) if gas_prices else 0
            }
        except Exception as e:
            logger.error(f"Error calculating gas statistics: {str(e)}")
            return {}
