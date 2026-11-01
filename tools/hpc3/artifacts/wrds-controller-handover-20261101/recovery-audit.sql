WITH expected(number,queue_id,request_sha,handover_sha,native_id,record_sha,replacement_sha) AS (VALUES
('1','ecd1aff4-21bb-4920-a95e-78e8775642d8','4244ff35f2064b4905e08745086a46f849526164616d6f12066778370b280de0','beb3c0159732573f1d56628ce6db33a846d238c50065597a180a263da3317497','58708603','649889fe71795fd20bafd5a6b08bdecce711d7a5d28811fe6eb85a7ea262869c','e75e1383dfc9a9836ae344b51c8fe9ff0d851619572c6261bd3fc0cc1f9552f8'),
('2','f2a5c21c-2e7e-4c30-8109-f640aa1d054c','5b656ced3ccabe709c33a7576eadf81da00f4ed884ce27b18af1eaae7c1cdd1e','732326e2b0a39cef6f5ab75737b3da03a1ce49d27f170860c00d012dab4f0e95','58708606','d41dae205d60a9a7113744cf026db7369ef309649dcaf77575b073e0f2be052f','e0a882bd5a2b425f31299741c96d13f7af0b271b2bbd55d4149e2fd3171e74c8'),
('3','490bc5e5-88fc-454d-8f9c-0f11a8cc8630','46f578777eebb9ff69c4b1d28db2291f7b17838b7f159a24d453d88894b0279d','c7697d8d9752637e1698e9f045b19c79cdfa3d28c37ca7511cb8073ad8206fcd','58708609','798dcf7a0b7a982724e1047a0fbc648afa280e3374f03a3c8c27a392093faf39','f4e6dbf95c74d461fc12903c512e61140d97cf51e1585e6a7025162c1cb87050'),
('4','320b4b70-a5e3-4c84-8e91-c37ed099d7ca','e4cc89392eb4f0cc5641e53dbdf1e97b61e4e789ac2691a32d6a098ba83cc3ba','56eadb9d8568fecbd159570a7de90264badc474e5247f348155fc73288a6a3ef','58708611','6d973877b4ef8f6a8ed13a2a1f710721bfc0e4434ffbc4073d0de8ddad4926c8','c3f78f84a8c4e403234c20ec3f06e2873b2b10176fcbfdf5a6ae4010ff6ed198')
), handovers AS (
 SELECT created_at,substring(body from position(chr(10) in body)+1)::jsonb AS doc
 FROM agent_task_updates
 WHERE task_id='a9b2927c-3775-46cc-b86e-2aeb0c2c4479'
 AND author='hpc-report-a9b2927c'
 AND body LIKE 'RESEARCH HANDOVER LOAD-REPORT%'
 AND created_at>'2026-11-01T10:09:00Z'
), native AS (
 SELECT created_at,substring(body from position(chr(10) in body)+1)::json AS doc
 FROM agent_task_updates
 WHERE task_id='a9b2927c-3775-46cc-b86e-2aeb0c2c4479'
 AND author='hpc-report-a9b2927c'
 AND body LIKE 'HPC3 ATTEMPT LOAD-REPORT%'
 AND created_at>'2026-11-01T10:09:00Z'
), jobs AS (
 SELECT j.*,substring(j.detail from position('{"state"' in j.detail))::jsonb AS outcome
 FROM fleet_dispatch_jobs j JOIN expected e ON j.id::text=e.queue_id
 WHERE j.task_id='a9b2927c-3775-46cc-b86e-2aeb0c2c4479'
 AND j.status='passed' AND j.exit_code=0
 AND j.command='research' AND j.project='research/hpc3'
 AND j.claimed_by='fleet-research-diphtheria'
), matched AS (
 SELECT e.queue_id FROM expected e JOIN jobs j ON j.id::text=e.queue_id
 WHERE j.research_request->>'specSha256'=e.request_sha
 AND j.outcome->>'handover_sha256'=e.handover_sha
 AND j.outcome->>'native_job_id'=e.native_id
 AND j.outcome->>'record_sha256'=e.record_sha
 AND j.outcome->>'controller_python'='/home/corvis/research-control/wrds-queue-b53b0ea34/bin/python'
 AND EXISTS (
  SELECT 1 FROM handovers h
  WHERE h.created_at BETWEEN j.claimed_at AND j.closed_at
  AND h.doc->>'kind'='RESEARCH_HANDOVER_EVIDENCE'
  AND h.doc->>'request_sha256'=e.request_sha
  AND h.doc->>'replacement_sha256'=e.replacement_sha
  AND h.doc->>'handover_sha256'=e.handover_sha
  AND h.doc->>'original_python'='/home/corvis/research-control/wrds-selection-report-9b6cbf367/bin/python'
  AND h.doc#>>'{native_observation,state}'='complete'
  AND h.doc#>>'{native_observation,native_job_id}'=e.native_id
  AND h.doc#>>'{native_observation,record_sha256}'=e.record_sha
  AND h.doc#>>'{replacement_controller,controller_python}'='/home/corvis/research-control/wrds-selection-recovery-f38df09fe/bin/python'
  AND h.doc#>>'{replacement_controller,source_commit}'='f38df09fe86566f5201bc59f9f28968276e72423'
  AND h.doc#>>'{replacement_controller,provenance_sha256}'='f6c8fac74af41e674d14a6d1050bed0912efedbc05e3e91d0db4c293f1583d18'
  AND h.doc#>>'{replacement_controller,installed_files}'='584'
  AND h.doc#>>'{replacement_controller,selfcheck}'='image self-check OK'
  AND h.doc#>>'{replacement_controller,dependency_check}'='No broken requirements found.'
 ) AND EXISTS (
  SELECT 1 FROM native n
  WHERE n.created_at BETWEEN j.claimed_at AND j.closed_at
  AND n.doc->>'kind'='HPC3_ATTEMPT_EVIDENCE'
  AND n.doc#>>'{ledger,job_id}'=e.native_id
  AND n.doc#>>'{ledger,experiment,research_attempt}'='c28c7c78284a67970c4989dbc1e95048280a71e694722718d4013e25765afe1e:'||e.number
  AND n.doc#>>'{ledger,artifact}'='/data/homezvol1/wagnera3/research/selection-report-9b6cbf367/attempts/c28c7c78284a67970c4989dbc1e95048280a71e694722718d4013e25765afe1e/'||e.number||'/result/evaluation.json'
  AND (n.doc#>>'{ledger,submitted_at}')::timestamptz<'2026-11-01T02:00:00Z'
  AND n.doc#>>'{accounting,job,state}'='COMPLETED'
  AND n.doc#>>'{accounting,exit_code}'='0:0'
  AND n.doc#>>'{measurement,dataset,bundle_sha256}'='e0b1f3b799d1a52ab79c0073e00d9b648ad1b042a70f334405e4a31e69f07b14'
  AND encode(sha256(convert_to((n.doc->'record')::text,'UTF8')),'hex')=e.record_sha
 )
)
SELECT count(DISTINCT queue_id) FROM matched;
