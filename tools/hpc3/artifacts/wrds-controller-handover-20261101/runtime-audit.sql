WITH reports AS (
 SELECT substring(body from position(chr(10) in body)+1)::jsonb AS doc
 FROM agent_task_updates
 WHERE id='6322f701-0787-49eb-90a1-8703898de4e8'
 AND task_id='04c0cdbe-278f-41bb-8141-a170fc0e5324'
 AND author='hpc-report-04c0cdbe'
 AND body LIKE 'HPC3 CONTROLLER LOAD-REPORT%'
)
SELECT count(*) FROM reports
WHERE doc->>'kind'='HPC3_CONTROLLER_EVIDENCE'
AND doc->>'source_commit'='b53b0ea343de96df607ad7dbc877c07ec5220668'
AND doc->>'controller_python'='/home/corvis/research-control/wrds-queue-b53b0ea34/bin/python'
AND doc->>'controller_python_sha256'='ca969be9adfbddc0a593745da84da0348a4c2baf849daae18806a344708934c4'
AND doc->>'provenance_sha256'='8d8a6969f7e53a5d2a20cafce664e0a04c3e4d4b139584325a884b2360381f86'
AND doc#>>'{verified_wheels,fleet-0.1.0-py3-none-any.whl,sha256}'='6b2a87978df884d9834ece89d78137cdc3fdab02effc56b482d8b6be421c9e9e'
AND (SELECT count(*) FROM jsonb_object_keys(doc->'verified_wheels'))=5
AND doc->>'installed_files'='485'
AND doc->>'selfcheck_sha256'='69d544204b2eb0cf20b2d4a159938c92dcaefd4bfdf2ee0f9c2f3a4b33ad292e'
AND doc->>'selfcheck'='image self-check OK'
AND doc->>'dependency_check'='No broken requirements found.';
