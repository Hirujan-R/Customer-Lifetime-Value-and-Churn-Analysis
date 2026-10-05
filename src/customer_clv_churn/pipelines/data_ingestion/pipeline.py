"""Pipeline for ingesting and cleaning the UCI Online Retail II dataset."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    clean_transactions,
    download_online_retail_ii,
    load_raw_transactions,
    summarise_transactions,
    validate_transactions,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the ``data_ingestion`` pipeline."""
    return pipeline(
        [
            node(
                func=download_online_retail_ii,
                inputs=[
                    "params:data_ingestion.source_urls",
                    "params:data_ingestion.raw_dir",
                    "params:data_ingestion.raw_filename",
                    "params:data_ingestion.force_download",
                    "params:data_ingestion.download_timeout_seconds",
                ],
                outputs="raw_retail_download_path",
                name="download_online_retail_ii_node",
            ),
            node(
                func=load_raw_transactions,
                inputs=["raw_retail_download_path", "params:data_ingestion.sheet_names"],
                outputs="transactions_raw",
                name="load_raw_transactions_node",
            ),
            node(
                func=clean_transactions,
                inputs=["transactions_raw", "params:data_ingestion.cleaning"],
                outputs="transactions",
                name="clean_transactions_node",
            ),
            node(
                func=validate_transactions,
                inputs=["transactions", "params:data_ingestion.expected"],
                outputs="data_quality_report",
                name="validate_transactions_node",
            ),
            node(
                func=summarise_transactions,
                inputs="transactions",
                outputs="transactions_summary",
                name="summarise_transactions_node",
            ),
        ]
    )
