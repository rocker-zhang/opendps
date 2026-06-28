"""Entry point: python -m opendps.operator"""
import kopf
import logging
logging.basicConfig(level=logging.INFO)
kopf.run()
