# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import datetime
import os
import sys

import pandas as pd
from azure.kusto.data import KustoConnectionStringBuilder
from azure.kusto.data.data_format import DataFormat
from azure.kusto.ingest import IngestionProperties, QueuedIngestClient, ReportLevel


def parse_arguments():
    """
    Parses command-line arguments and returns an object with each argument as a field.

    :return: An object whose fields represent the parsed command-line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--workspace", help="Path to the local csv", required=True)
    parser.add_argument("-c", "--commit_hash", help="Commit hash", required=True)
    parser.add_argument(
        "-d",
        "--commit_datetime",
        help="Commit datetime in Python's datetime ISO 8601 format",
        required=True,
        type=datetime.datetime.fromisoformat,
    )
    parser.add_argument("-t", "--trt_version", help="Tensorrt Version", required=True)
    parser.add_argument("-b", "--branch", help="Branch", required=True)
    parser.add_argument("--kusto_conn", help="Kusto connection URL", required=True)
    parser.add_argument("--database", help="Database name", required=True)
    return parser.parse_args()


def write_table(
    ingest_client, database_name, table, table_name, upload_time, identifier, branch, commit_id, commit_date
):
    """
    Uploads the provided table to the database. This function also appends the upload time and unique run identifier
    to the table.

    :param ingest_client: An instance of QueuedIngestClient used to initiate data ingestion.
    :param table: The Pandas table to ingest.
    :param table_name: The name of the table in the database.
    :param upload_time: A datetime object denoting the data's upload time.
    :param identifier: An identifier that associates the uploaded data with an ORT commit/date/branch.
    """

    if table.empty:
        return

    # Add upload time and identifier columns to data table.
    table = table.assign(UploadTime=str(upload_time))
    table = table.assign(Identifier=identifier)
    table = table.assign(Branch=branch)
    table = table.assign(CommitId=commit_id)
    table = table.assign(CommitDate=str(commit_date))
    ingestion_props = IngestionProperties(
        database=database_name,
        table=table_name,
        data_format=DataFormat.CSV,
        report_level=ReportLevel.FailuresAndSuccesses,
    )
    # append rows
    ingest_client.ingest_from_dataframe(table, ingestion_properties=ingestion_props)


def get_identifier(commit_datetime, commit_hash, trt_version, branch):
    """
    Returns an identifier that associates uploaded data with an ORT commit/date/branch and a TensorRT version.

    :param commit_datetime: The datetime of the ORT commit used to run the benchmarks.
    :param commit_hash: The hash of the ORT commit used to run the benchmarks.
    :param trt_version: The TensorRT version used to run the benchmarks.
    :param branch: The name of the ORT branch used to run the benchmarks.

    :return: A string identifier.
    """

    date = str(commit_datetime.date())  # extract date only
    return date + "_" + commit_hash + "_" + trt_version + "_" + branch


def main():
    """
    Entry point of this script. Uploads data produced by benchmarking scripts to the database.
    """

    args = parse_arguments()

    # connect to database
    kcsb_ingest = KustoConnectionStringBuilder.with_az_cli_authentication(args.kusto_conn)
    ingest_client = QueuedIngestClient(kcsb_ingest)
    identifier = get_identifier(args.commit_datetime, args.commit_hash, args.trt_version, args.branch)
    upload_time = datetime.datetime.now(tz=datetime.timezone.utc).replace(microsecond=0)

    try:
        workspace = args.workspace
        csv_filenames = os.listdir(workspace)
        os.chdir(workspace)
        
        tables = [
            "ep_valgrind_record"
        ]

        table_results = {}
        for table_name in tables:
            table_results[table_name] = pd.DataFrame()

        # Parse csv
        for csv in csv_filenames:
            table = pd.read_csv(csv)
            if "ep_valgrind_record" in csv:
                table_results["ep_valgrind_record"] = pd.concat(
                    [table_results["ep_valgrind_record"], table], ignore_index=True
                )

        for table in tables:
            print("writing " + table + " to database")
            write_table(
                ingest_client,
                args.database,
                table_results[table],
                table,
                upload_time,
                identifier,
                args.branch,
                args.commit_hash,
                args.commit_datetime,
            )

    except BaseException as e:
        print(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()