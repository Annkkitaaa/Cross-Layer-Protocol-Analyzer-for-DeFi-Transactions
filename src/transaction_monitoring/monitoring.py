
import asyncio
from typing import Dict, List, Callable
import json
import logging
from datetime import datetime
from .cryptography import verify_signature, generate_zk_proof
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransactionMonitor:
    def __init__(self, eth_extractor, stark_extractor):
        """Initialize transaction monitor with both layer extractors."""
        self.eth_extractor = eth_extractor
        self.stark_extractor = stark_extractor
        self.logger = logger
        self.callbacks: List[Callable] = []
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def start_monitoring(self):
        """Start monitoring transactions on both layers."""
        await asyncio.gather(
            self.monitor_ethereum(),
            self.monitor_starknet()
        )

    def add_callback(self, callback: Callable):
        """Add callback for transaction events."""
        self.callbacks.append(callback)

    async def monitor_ethereum(self):
        """Monitor Ethereum transactions."""
        async def process_transaction(tx):
            try:
                # Verify transaction signature
                is_valid = await self.executor.submit(
                    verify_signature,
                    tx['from'],
                    tx['hash']
                )

                if not is_valid:
                    self.logger.warning(f"Invalid signature for transaction {tx['hash']}")
                    return

                # Generate ZK proof for transaction validation
                proof = await self.executor.submit(
                    generate_zk_proof,
                    tx
                )

                enriched_tx = {
                    **tx,
                    'layer': 'ethereum',
                    'verification': {
                        'signature_valid': is_valid,
                        'zk_proof': proof
                    },
                    'timestamp': datetime.now().timestamp()
                }

                # Notify all callbacks
                for callback in self.callbacks:
                    await callback(enriched_tx)

            except Exception as e:
                self.logger.error(f"Error processing Ethereum transaction: {str(e)}")

        await self.eth_extractor.monitor_pending_transactions(process_transaction)

    async def monitor_starknet(self):
        """Monitor Starknet transactions."""
        async def process_block(block):
            try:
                for tx_hash in block.transaction_hashes:
                    tx = await self.stark_extractor.get_transaction(tx_hash)
                    if tx:
                        # Generate proof for Starknet transaction
                        proof = await self.executor.submit(
                            generate_zk_proof,
                            tx
                        )

                        enriched_tx = {
                            **tx,
                            'layer': 'starknet',
                            'verification': {
                                'zk_proof': proof
                            },
                            'timestamp': datetime.now().timestamp()
                        }

                        # Notify all callbacks
                        for callback in self.callbacks:
                            await callback(enriched_tx)

            except Exception as e:
                self.logger.error(f"Error processing Starknet block: {str(e)}")

        await self.stark_extractor.monitor_blocks(process_block)

    async def analyze_cross_layer_transaction(self, eth_tx_hash: str, stark_tx_hash: str) -> Dict:
        """Analyze related transactions across both layers."""
        eth_tx = await self.eth_extractor.get_transaction(eth_tx_hash)
        stark_tx = await self.stark_extractor.get_transaction(stark_tx_hash)

        if not eth_tx or not stark_tx:
            raise ValueError("One or both transactions not found")

        return {
            'ethereum': eth_tx,
            'starknet': stark_tx,
            'analysis': {
                'time_difference': abs(eth_tx['timestamp'] - stark_tx['timestamp']),
                'eth_gas_cost': eth_tx['gas_price'] * eth_tx['gas_used'],
                'stark_fee': stark_tx['actual_fee'],
                'timestamp': datetime.now().timestamp()
            }
        }