"""
src/quality.py — Data quality validation via Great Expectations.

Flow:
  run_quality_checks(df, config) → summary dict
  save_quality_report(summary, output_dir) → Path of saved JSON
"""
import json
import pandas as pd
from typing import Any
from pathlib import Path
from datetime import datetime
from src.utils.logger import get_logger

def _import_ge():
    """Import great_expectations; raise a descriptive ImportError if missing."""
    try:
        import great_expectations as gx
        import great_expectations.expectations as gxe
        return gx, gxe
    except ImportError as exc:
        raise ImportError(
            "great-expectations not found.\n"
            "Install with:  pip install great-expectations>=1.0.0\n"
            "Or add it to requirements.txt and run pip install -r requirements.txt"
        ) from exc
    
def _build_ephemeral_context(df: pd.DataFrame, suite_name: str):
    """
    Create an ephemeral GE context with a pandas datasource and return
    (context, batch_definition, suite).

    The ephemeral context does not persist anything to disk — ideal for
    reproducible pipelines. All configuration comes from YAML, not GE files.
    """
    gx, _ =_import_ge()

    # ephemeral context
    context = gx.get_context(
        mode='ephemeral'
    )

    # pandas datasource
    data_source = context.data_sources.add_pandas("pipeline_source")
    asset = data_source.add_dataframe_asset(
        name="input_data"
    )
    batch_def = asset.add_batch_definition_whole_dataframe('full_batch')

    # create an empty suite (expectations will be added dynamically)
    suite = context.suites.add(
        gx.ExpectationSuite(name=suite_name)
    )

    return context, batch_def, suite

def _snake_to_pascal(snake_str: str) -> str:
    """Convert snake_case to PascalCase (e.g.: row_count → RowCount)."""
    return ''.join(word.capitalize() for word in snake_str.split('_'))

def _resolve_expectation_class(gxe, type_name: str):
    """
    Resolve a GE class from a snake_case or PascalCase name.

    Accepts both conventions so the YAML user does not have to memorize
    exact capitalization.

    Raises:
        AttributeError: if the type does not exist in the GE expectations module.
    """
    # try PascalCase first
    pascal = _snake_to_pascal(type_name)

    if hasattr(gxe, pascal):
        return getattr(gxe, pascal)
    elif hasattr(gxe, type_name):
        return getattr(gxe, type_name)
    else:
        raise AttributeError(
            f"Expectation type '{type_name}' not found in great_expectations.expectations module."
        )

def _populate_suite_with_expectations(
    suite,
    table_expectations: list[dict[str, Any]],
    column_expectations: dict[str, list[dict[str, Any]]],
    gxe
) -> None:
    """
    Add expectations dynamically to the suite from YAML lists.

    Strategy:
      - table_exps: expectations without a column (e.g.: row count, schema)
      - column_exps: dict column → list of expectations

    Dynamic dispatch via _resolve_expectation_class makes this module
    independent of which specific expectations the YAML defines.
    """
    # table expectations
    for exp in (table_expectations or []):
        exp_class = _resolve_expectation_class(gxe, exp['type'])
        kwargs = exp.get('kwargs', {})
        suite.add_expectation(
            exp_class(**kwargs)
        )

    # column expectations
    for col, exps in (column_expectations or {}).items():
        for exp in exps:
            exp_class = _resolve_expectation_class(gxe, exp['type'])
            kwargs = exp.get('kwargs', {})
            suite.add_expectation(
                exp_class(column=col, **kwargs)
            )

# ----------------------------------------------------------------------------------------------------------------    
# main function
def run_quality_checks(
    df: pd.DataFrame,
    config: dict[str, Any],
    logging_config: dict[str, Any]
) -> dict[str, Any]:
    """
    Execute all quality checks defined in the config.

    Args:
        df:             DataFrame to validate (output of ingestion).
        config:         Combined dictionary (pipeline.yaml + quality.yaml).
        logging_config: 'logging' section from pipeline.yaml.

    Returns:
        Dictionary with:
          success  (bool)   — True if all expectations passed
          total    (int)    — total number of checks
          passed   (int)    — checks that passed
          failed   (int)    — checks that failed
          results  (object) — raw GE result (for save_quality_report)

    Raises:
        RuntimeError: if fail_pipeline_on_error=true and any check fails.
        ImportError:  if great-expectations is not installed.
    """
    gx, gxe = _import_ge()
    logger = get_logger('quality_checks_module', logging_config=logging_config)

    # read quality parameters from config
    suite_name = config.get('suite_name', 'default_suite')
    fail_on_error = config.get('fail_pipeline_on_error', True)
    table_expectations = config.get('table_expectations', [])
    column_expectations = config.get('column_expectations', [])

    total_exp = len(table_expectations) + sum(len(v) for v in column_expectations.values())
    logger.info('Starting quality checks: %d checks defined', total_exp)

    # build ephemeral context and suite
    context, batch_def, suite = _build_ephemeral_context(df, suite_name)

    # populate suite with config expectations
    _populate_suite_with_expectations(
        suite,
        table_expectations,
        column_expectations,
        gxe
    )

    # create and register the ValidationDefinition
    validation_def = context.validation_definitions.add(
        gx.ValidationDefinition(
            name=f"{suite_name}_validation",
            data=batch_def,
            suite=suite
        )
    )

    # run validation
    logger.info('Running quality validation...')
    results = validation_def.run(
        batch_parameters={
            'dataframe': df
        }
    )

    # summarize results
    success = results.success
    total = len(results.results)
    passed = sum(1 for r in results.results if r.success)
    failed = total - passed

    logger.info('-'*60)
    logger.info('Quality validation results:')
    logger.info('  Total checks: %d', total)
    logger.info('  Passed checks: %d', passed)
    logger.info('  Failed checks: %d', failed)
    logger.info('-'*60)

    # detailed log per expectation
    for r in results.results:
        status   = "OK  " if r.success else "FAIL"
        exp_type = r.expectation_config.type
        col      = r.expectation_config.kwargs.get("column", "(table)")
        logger.info("  [%s] %-50s  col=%-25s", status, exp_type, col)

    # explicit failure if configured — production behavior
    if fail_on_error and not success:
        raise RuntimeError(
            f"Data quality FAILED: {failed}/{total} checks failed.\n"
            "Review quality.yaml or inspect the input data.\n"
            "To continue despite failures, set fail_pipeline_on_error: false"
        )

    return {
        "success": success,
        "total":   total,
        "passed":  passed,
        "failed":  failed,
        "results": results,
    }

def save_quality_report(
    summary: dict[str, Any],
    output_dir: Path,
    logging_config: dict[str, Any] | None = None,
) -> Path:
    """
    Serialize the quality summary to readable JSON.

    Only the summary is serialized — the GE results object is not
    directly JSON-serializable. Information from each check is extracted
    and normalized here.

    Args:
        summary:        Return value from run_quality_checks().
        output_dir:     Directory where the report will be saved.
        logging_config: 'logging' section from pipeline.yaml.

    Returns:
        Path to the generated JSON file.
    """
    logger = get_logger("quality", logging_config or {})
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"quality_report_{timestamp}.json"

    details = []
    for r in summary["results"].results:
        # Extract kwargs without the 'column' key to avoid duplication
        raw_kwargs = dict(r.expectation_config.kwargs)
        column     = raw_kwargs.pop("column", None)

        # result can contain observed statistics (e.g.: % nulls, count)
        raw_result = r.result if isinstance(r.result, dict) else {}

        details.append({
            "type":    r.expectation_config.type,
            "column":  column,
            "success": r.success,
            "kwargs":  raw_kwargs,
            "result":  {k: v for k, v in raw_result.items()
                        if not isinstance(v, (list, dict)) or len(str(v)) < 500},
        })

    report = {
        "success": summary["success"],
        "total":   summary["total"],
        "passed":  summary["passed"],
        "failed":  summary["failed"],
        "details": details,
    }

    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=str)

    logger.info("Quality report saved: %s", report_path)
    return report_path