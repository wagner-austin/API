WITH reports AS (
 SELECT id, created_at, substring(body from position(chr(10) in body)+1)::jsonb AS doc
 FROM agent_task_updates
 WHERE task_id='42a8d507-0c77-44fd-99a1-a37aedde2f1e'
 AND author='hpc-report-42a8d507'
 AND id IN ('712dca27-d5c6-417a-b3e6-e473ebd0501b','22dfef1e-b1ae-4299-8047-44dea4d14e68','47471e03-4f43-4353-8722-071d5fbf5136','32e4e280-5d1b-46bb-b3e1-10c1c2f5a7c7')
), controller AS (
 SELECT * FROM reports WHERE id='712dca27-d5c6-417a-b3e6-e473ebd0501b'
 AND doc->>'kind'='HPC3_CONTROLLER_EVIDENCE'
 AND doc->>'source_commit'='11c0bdbdd167d953d5464f9367242370f51c5765'
 AND doc->>'controller_python'='/home/corvis/research-control/wrds-resources-11c0bdbdd/bin/python'
 AND doc->>'provenance_sha256'='c6fa9e7d01bbc4a4d60732cbd9ab10de5cb33d932e485e6fed6a90d4014589fb'
 AND doc#>>'{verified_wheels,covenant_ml-0.1.0-py3-none-any.whl,sha256}'='af12220e30cad45462970ed375029bdaa69442ffc3772043fbdcd55f1b970aec'
 AND doc#>>'{verified_wheels,hpc3-0.1.0-py3-none-any.whl,sha256}'='760ad4644a21a9011a1a739c8ea660f1fd65170192a7f35fab627b02fab162dd'
 AND doc->>'installed_files'='584'
 AND doc->>'selfcheck'='image self-check OK'
 AND doc->>'dependency_check'='No broken requirements found.'
), image AS (
 SELECT * FROM reports WHERE id='22dfef1e-b1ae-4299-8047-44dea4d14e68'
 AND doc->>'kind'='HPC3_IMAGE_BUILD_EVIDENCE'
 AND doc#>>'{accounting,job,job_id}'='58719730'
 AND doc#>>'{accounting,job,state}'='COMPLETED'
 AND doc#>>'{accounting,exit_code}'='0:0'
 AND doc->>'selfcheck_passes'='2'
 AND doc->>'dependency_check'='passed'
 AND (SELECT count(*) FROM jsonb_object_keys(doc->'verified_files'))=14
 AND doc#>>'{verified_files,/data/homezvol1/wagnera3/research-images/resources-11c0bdbdd/research-resources.sif}'='39c0129fff1c6c449e81e6a3f822f75864e21650efa566aeb27346546e66abd1'
 AND doc#>>'{verified_files,/data/homezvol1/wagnera3/research-images/resources-11c0bdbdd/wheels/covenant_ml-0.1.0-py3-none-any.whl}'='af12220e30cad45462970ed375029bdaa69442ffc3772043fbdcd55f1b970aec'
), stage AS (
 SELECT * FROM reports WHERE id='47471e03-4f43-4353-8722-071d5fbf5136'
 AND doc->>'kind'='HPC3_STAGE_EVIDENCE'
 AND doc->>'host'='hpc3'
 AND doc->>'verified_files'='2'
 AND doc#>>'{manifest,provenance,policy}'='bbfad34870cf4d961976baff902c574d49c87da12474cadc4a6638b71322555a'
 AND doc#>>'{manifest,files,0,name}'='request.json'
 AND doc#>>'{manifest,files,0,sha256}'='d0ba260478df2a1bea793a78e8fba0da334f50c39dac1cc4f648e9cb1003bd5d'
), preflight AS (
 SELECT * FROM reports WHERE id='32e4e280-5d1b-46bb-b3e1-10c1c2f5a7c7'
 AND doc->>'kind'='HPC3_PREFLIGHT_EVIDENCE'
 AND doc->>'host'='hpc3'
 AND doc->>'queued'='false'
 AND doc->>'reporter_sha256'='a712064d3841314fc12e193a66672d589b0f5b05fae1438cd7e354eae6b129e4'
 AND jsonb_array_length(doc->'admissions')=1
 AND doc#>>'{admissions,0,spec,cpus}'='4'
 AND doc#>>'{admissions,0,spec,mem_gb}'='128'
 AND doc#>>'{admissions,0,spec,minutes}'='121'
 AND doc#>>'{admissions,0,spec,partition}'='free'
 AND doc#>>'{admissions,0,spec,deterministic}'='true'
 AND doc#>>'{admissions,0,spec,experiment,research_attempt}'='bbfad34870cf4d961976baff902c574d49c87da12474cadc4a6638b71322555a:1'
 AND doc#>>'{admissions,0,spec,experiment,request_sha256}'='d0ba260478df2a1bea793a78e8fba0da334f50c39dac1cc4f648e9cb1003bd5d'
 AND doc#>>'{admissions,0,spec,image,sha256}'='39c0129fff1c6c449e81e6a3f822f75864e21650efa566aeb27346546e66abd1'
 AND doc#>>'{admissions,0,verdict,processors}'='4'
 AND doc#>>'{admissions,0,verdict,partition}'='free'
 AND doc#>>'{admissions,0,verdict,node_list}'='hpc3-14-31'
 AND doc#>>'{admissions,0,image_drift,image_commit}'='11c0bdbdd167d953d5464f9367242370f51c5765'
)
SELECT count(*) FROM controller, image, stage, preflight
WHERE controller.created_at < image.created_at
AND image.created_at < stage.created_at
AND stage.created_at < preflight.created_at;
