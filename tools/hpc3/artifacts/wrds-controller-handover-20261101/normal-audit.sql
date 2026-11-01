WITH job AS (
 SELECT j.*,substring(j.detail from position('{"state"' in j.detail))::json AS outcome
 FROM fleet_dispatch_jobs j
 WHERE j.id='eecf34ae-f66a-4b6a-a3ab-3f0f108d87c0'
 AND j.status='passed' AND j.exit_code=0
 AND j.task_id='a9b2927c-3775-46cc-b86e-2aeb0c2c4479'
 AND j.project='research/hpc3' AND j.command='research'
 AND j.claimed_by='fleet-research-diphtheria'
 AND j.research_request->>'specSha256'='ad15cc91bf41c01a5feb3a73cf90f2ef52ce19def6a4a1b2673e9d7567512082'
), native AS (
 SELECT created_at,substring(body from position(chr(10) in body)+1)::json AS doc
 FROM agent_task_updates
 WHERE task_id='a9b2927c-3775-46cc-b86e-2aeb0c2c4479'
 AND author='hpc-report-a9b2927c'
 AND body LIKE 'HPC3 ATTEMPT LOAD-REPORT%'
 AND created_at>'2026-11-01T10:13:00Z'
)
SELECT count(*) FROM job j
WHERE j.outcome->>'state'='complete'
AND j.outcome->>'controller_python'='/home/corvis/research-control/wrds-queue-b53b0ea34/bin/python'
AND (j.outcome->'handover_sha256')::jsonb='null'::jsonb
AND (SELECT count(*) FROM fleet_dispatch_jobs r
     WHERE r.research_request->>'specSha256'='ad15cc91bf41c01a5feb3a73cf90f2ef52ce19def6a4a1b2673e9d7567512082')=1
AND EXISTS (
 SELECT 1 FROM native n
 WHERE n.created_at BETWEEN j.claimed_at AND j.closed_at
 AND n.doc->>'kind'='HPC3_ATTEMPT_EVIDENCE'
 AND n.doc#>>'{ledger,job_id}'=j.outcome->>'native_job_id'
 AND n.doc#>>'{ledger,experiment,research_attempt}'='c28c7c78284a67970c4989dbc1e95048280a71e694722718d4013e25765afe1e:5'
 AND n.doc#>>'{ledger,experiment,request_sha256}'='4b5dbd2c26a46866f67d3ba5092129b1ed7f0e3ef5e3ac3cdcba7a3a738cfa66'
 AND n.doc#>>'{ledger,artifact}'='/data/homezvol1/wagnera3/research/selection-report-9b6cbf367/attempts/c28c7c78284a67970c4989dbc1e95048280a71e694722718d4013e25765afe1e/5/result/evaluation.json'
 AND n.doc#>>'{accounting,job,state}'='COMPLETED'
 AND n.doc#>>'{accounting,exit_code}'='0:0'
 AND n.doc#>>'{measurement,dataset,bundle_sha256}'='e0b1f3b799d1a52ab79c0073e00d9b648ad1b042a70f334405e4a31e69f07b14'
 AND encode(sha256(convert_to((n.doc->'record')::text,'UTF8')),'hex')=j.outcome->>'record_sha256'
);
