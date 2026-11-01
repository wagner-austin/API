WITH expected(name,id,attempt,seed,code,request_sha) AS (VALUES
('wrong_candidate','e2831b84-b179-499e-b851-cc48781346ff','6','11','2','7c6f8d0497a2928d3abb03c1d29d5001b47d34899c94af0be7b449f5a2f91cba'),
('wrong_seed','a100f9a7-aecf-4b22-b07d-271bdeb7ec40','2','23','2','1dfcbc2afe071b7d657f391db6e9f32df9ceb4ad8e71d48b1e5df3ae04ced98b'),
('winner','baa53190-6da3-4141-9d2f-9f0ec60e6d8a','1','11','0','23c89dc6deed544013b7d2d507d2d8801f5fab022879216e1d75593821559eb0'),
('replay','62033c80-b373-45a7-87d6-ee2abd80db65','1','11','2','23c89dc6deed544013b7d2d507d2d8801f5fab022879216e1d75593821559eb0')
), measured AS (
 SELECT e.name,substring(u.body from position(chr(10) in u.body)+1)::jsonb AS doc,e.attempt,e.seed,e.code,e.request_sha
 FROM expected e JOIN agent_task_updates u ON u.id::text=e.id
 WHERE u.task_id='a9b2927c-3775-46cc-b86e-2aeb0c2c4479' AND u.author='hpc-report-a9b2927c'
 AND u.body LIKE 'RESEARCH SELECTION LOAD-REPORT%'
), valid AS (
 SELECT name,doc FROM measured
 WHERE doc->>'kind'='RESEARCH_SELECTION_EVIDENCE'
 AND doc->>'controller_python'='/home/corvis/research-control/wrds-selection-recovery-f38df09fe/bin/python'
 AND doc->>'controller_python_sha256'='ca969be9adfbddc0a593745da84da0348a4c2baf849daae18806a344708934c4'
 AND doc->>'reporter_sha256'='1c88e60d02d591f72e1a3eb223350265f2a4dbfd20ca114b28fb925e7add32b8'
 AND doc->>'policy_digest'='c28c7c78284a67970c4989dbc1e95048280a71e694722718d4013e25765afe1e'
 AND doc->>'ledger'='/home/corvis/research-data/wrds/experiments/selection-report-9b6cbf367/attempts.sqlite'
 AND doc->>'final_manifest_digest'='696a9726d90451eff2d97510e6aa85cdd3349d02b19fb9583cb938cc0718decf'
 AND doc->>'attempt'=attempt AND doc->>'seed'=seed
 AND doc->>'exit_status'=code AND doc->>'request_sha256'=request_sha
 AND (doc->>'started_at')::timestamptz <= (doc->>'finished_at')::timestamptz
), chain AS (
 SELECT 1 FROM valid c,valid s,valid w,valid r
 WHERE c.name='wrong_candidate' AND s.name='wrong_seed' AND w.name='winner' AND r.name='replay'
 AND c.doc->'selected'='[]'::jsonb AND s.doc->'selected'='[]'::jsonb
 AND c.doc->'messages'='["usage: RESEARCH_SELECTION_WINNER: require the winner''s fixed final seed"]'::jsonb
 AND s.doc->'messages'=c.doc->'messages'
 AND c.doc->>'journal_sha256_before'=c.doc->>'journal_sha256_after'
 AND s.doc->>'journal_sha256_before'=s.doc->>'journal_sha256_after'
 AND s.doc->>'journal_sha256_before'=c.doc->>'journal_sha256_after'
 AND w.doc->>'journal_sha256_before'=s.doc->>'journal_sha256_after'
 AND w.doc->>'journal_sha256_before'<>w.doc->>'journal_sha256_after'
 AND w.doc->'messages'='[]'::jsonb
 AND w.doc->'selected'='[{"number":1,"candidate":"88f5b90aa93f70e0f026f7dbc5778c4c2f5984f401e4f74165c0864da5cbf345","seed":11,"selection":"selected","final_manifest_digest":"696a9726d90451eff2d97510e6aa85cdd3349d02b19fb9583cb938cc0718decf"}]'::jsonb
 AND r.doc->'selected'=w.doc->'selected'
 AND r.doc->>'journal_sha256_before'=w.doc->>'journal_sha256_after'
 AND r.doc->>'journal_sha256_before'=r.doc->>'journal_sha256_after'
 AND r.doc->'messages'='["usage: [RESEARCH-008] selection requires settled, unsealed validation"]'::jsonb
)
SELECT count(*) FROM chain;
