
from web3 import Web3
from typing import Dict, List, Optional
import os
import json
import asyncio
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EthereumDataExtractor:
    def __init__(self, infura_url: str):
        """Initialize Web3 connection with Infura."""
        self.w3 = Web3(Web3.HTTPProvider(infura_url))
        if not self.w3.is_connected():
            raise ConnectionError("Failed to connect to Ethereum network")
        self.logger = logger

    async def get_transaction(self, tx_hash: str) -> Optional[Dict]:
        """Fetch transaction details by hash."""
        try:
            tx = self.w3.eth.get_transaction(tx_hash)
            receipt = self.w3.eth.get_transaction_receipt(tx_hash)
            
            return {
                'hash': tx_hash,
                'from': tx['from'],
                'to': tx['to'],
                'value': self.w3.from_wei(tx['value'], 'ether'),
                'gas_price': self.w3.from_wei(tx['gasPrice'], 'gwei'),
                'gas_used': receipt['gasUsed'],
                'block_number': tx['blockNumber'],
                'timestamp': self.w3.eth.get_block(tx['blockNumber'])['timestamp'],
                'status': receipt['status']
            }
        except Exception as e:
            self.logger.error(f"Error fetching transaction {tx_hash}: {str(e)}")
            return None

    async def get_defi_protocol_events(self, contract_address: str, from_block: int) -> List[Dict]:
        """Fetch DeFi protocol events from a specific contract."""
        try:
            # Load ABI from JSON file (you'll need to provide the correct ABI)
            with open('data/raw/defi_abi.json', 'r') as f:
                contract_abi = json.load(f)

            contract = self.w3.eth.contract(
                address=self.w3.to_checksum_address(contract_address),
                abi=contract_abi
            )

            # Get all events
            events = contract.events.get_all_entries({
                'fromBlock': from_block,
                'toBlock': 'latest'
            })

            return [{
                'event_type': event['event'],
                'block_number': event['blockNumber'],
                'transaction_hash': event['transactionHash'].hex(),
                'args': dict(event['args']),
                'timestamp': self.w3.eth.get_block(event['blockNumber'])['timestamp']
            } for event in events]

        except Exception as e:
            self.logger.error(f"Error fetching events from contract {contract_address}: {str(e)}")
            return []

    async def monitor_pending_transactions(self, callback):
        """Monitor pending transactions in real-time."""
        async def handle_pending(transaction_hash):
            tx = await self.get_transaction(transaction_hash)
            if tx:
                await callback(tx)

        pending_filter = self.w3.eth.filter('pending')
        while True:
            for tx_hash in pending_filter.get_new_entries():
                await handle_pending(tx_hash)
            await asyncio.sleep(1)

    def get_gas_statistics(self, block_range: int = 100) -> Dict:
        """Calculate gas statistics from recent blocks."""
        latest_block = self.w3.eth.block_number
        gas_prices = []

        for block_number in range(latest_block - block_range, latest_block):
            block = self.w3.eth.get_block(block_number, True)
            for tx in block.transactions:
                gas_prices.append(self.w3.from_wei(tx['gasPrice'], 'gwei'))

        return {
            'average': sum(gas_prices) / len(gas_prices),
            'median': sorted(gas_prices)[len(gas_prices) // 2],
            'min': min(gas_prices),
            'max': max(gas_prices)
        }