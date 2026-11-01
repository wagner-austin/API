WITH report AS (
 SELECT substring(body from position(chr(10) in body)+1)::jsonb AS doc
 FROM agent_task_updates
 WHERE id='9d77722e-a7a3-4096-9815-01874194947a'
 AND task_id='a9b2927c-3775-46cc-b86e-2aeb0c2c4479' AND author='hpc-report-a9b2927c'
 AND body LIKE 'HPC3 CONTROLLER LOAD-REPORT%'
)
SELECT count(*) FROM report
WHERE doc->>'kind'='HPC3_CONTROLLER_EVIDENCE'
AND doc->>'source_commit'='f38df09fe86566f5201bc59f9f28968276e72423'
AND doc->>'controller_python'='/home/corvis/research-control/wrds-selection-recovery-f38df09fe/bin/python'
AND doc->>'controller_python_sha256'='ca969be9adfbddc0a593745da84da0348a4c2baf849daae18806a344708934c4'
AND doc->>'provenance_sha256'='f6c8fac74af41e674d14a6d1050bed0912efedbc05e3e91d0db4c293f1583d18'
AND doc#>>'{verified_wheels,covenant_ml-0.1.0-py3-none-any.whl,sha256}'='aa9abbdd47a80dcf73b3cdc3157f155708b7a3ee690e0e59c23c5ad36aa066ab'
AND (SELECT count(*) FROM jsonb_object_keys(doc->'verified_wheels'))=3
AND doc->>'installed_files'='584'
AND doc->>'selfcheck_sha256'='e69de9b66e326815fd6e0dfdd23b026b7513d800275cb950691234ee33d79f11'
AND doc->>'selfcheck'='image self-check OK'
AND doc->>'dependency_check'='No broken requirements found.';
