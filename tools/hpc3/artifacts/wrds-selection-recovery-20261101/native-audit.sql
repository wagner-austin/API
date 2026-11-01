WITH expected(number,id,spec_sha,native_id,record_sha,trial_sha) AS (VALUES
('1','ecd1aff4-21bb-4920-a95e-78e8775642d8','4244ff35f2064b4905e08745086a46f849526164616d6f12066778370b280de0','58708603','649889fe71795fd20bafd5a6b08bdecce711d7a5d28811fe6eb85a7ea262869c','23c89dc6deed544013b7d2d507d2d8801f5fab022879216e1d75593821559eb0'),
('2','f2a5c21c-2e7e-4c30-8109-f640aa1d054c','5b656ced3ccabe709c33a7576eadf81da00f4ed884ce27b18af1eaae7c1cdd1e','58708606','d41dae205d60a9a7113744cf026db7369ef309649dcaf77575b073e0f2be052f','1dfcbc2afe071b7d657f391db6e9f32df9ceb4ad8e71d48b1e5df3ae04ced98b'),
('3','490bc5e5-88fc-454d-8f9c-0f11a8cc8630','46f578777eebb9ff69c4b1d28db2291f7b17838b7f159a24d453d88894b0279d','58708609','798dcf7a0b7a982724e1047a0fbc648afa280e3374f03a3c8c27a392093faf39','a280705994e5a3d3a9482ca3c2ae055444f79d81649e8c86bab3058f6700e6fd'),
('4','320b4b70-a5e3-4c84-8e91-c37ed099d7ca','e4cc89392eb4f0cc5641e53dbdf1e97b61e4e789ac2691a32d6a098ba83cc3ba','58708611','6d973877b4ef8f6a8ed13a2a1f710721bfc0e4434ffbc4073d0de8ddad4926c8','8d8d95857c0001982c15d96fa9d4785341b85012cd00db6c8331e22beff6bd35'),
('5','eecf34ae-f66a-4b6a-a3ab-3f0f108d87c0','ad15cc91bf41c01a5feb3a73cf90f2ef52ce19def6a4a1b2673e9d7567512082','58719001','72effd337bd54fb26e1b9c478d4e5783f2e8f778be697ed1c3f6e97492be8afd','4b5dbd2c26a46866f67d3ba5092129b1ed7f0e3ef5e3ac3cdcba7a3a738cfa66'),
('6','73eefaff-e704-4f78-b67a-ae4a3824b25b','436992d303a39522db41a3116a32cf5f634028ba839a9d0c13be432ec126fb58','58719021','eaabcf0f38ea8237304b036e83941df9c693b538a051c29d79b153229a3f38b4','7c6f8d0497a2928d3abb03c1d29d5001b47d34899c94af0be7b449f5a2f91cba'),
('7','998bd825-a991-46cb-bcbc-d59b6da3a004','a3fff1185f1d96696fff70e5ac0c9d446b025e3b4d3e58923e4c8a659681f7ad','58719038','fd0cd378b312cef967d11fa42d579223ea9ccde21e3e24750277eec7711ea19a','84339b6981ab08b3a5e3dc95ee3144f96c9410c50530d08b162375e379b7e6dc'),
('8','ada40b76-c216-46c7-9e7d-3b04d99dcfe2','f34a0454dca5088466bce1f362ee6c8066ea0275d8e719702bfb33185f4d33f7','58719044','bceaa2975ee9a4090e03dcc37432f62b9c313b97e2f6adf4c187de79f1be421e','4a600dd903d9980d575ae1a126f445b53bf313ff229da996562840df4f5a0263'),
('9','fafc7edf-f8f0-4e83-a44e-4aae95bcc136','544fb81c6cf986fb8117004fe5dcd0b3c3570b9f956fa40d49d5fa7c6082ef4b','58719187','f34527a5cfd7602dd3ecfe8afb90824036815d40cf90fc987710a3bd5a0863d3','5f204d85670a1d0712709010dd4257329fee6533ea555a390b4184b54688d7a8'),
('10','cb0c2166-6334-4306-90ce-04cddba43b61','74531e4eba142b4e6bec04fbd37f3cf1babf4b7ccf64925afbdc96b5811c2c75','58719195','6ff3f1c46f55da5036ab8878da6b15b7333de26e54002dcd0cf24e031f53e807','baa91a1c6782325af922a740506383a2fa64c142fea0ec2d52d5dcd4d6d4f54c')
), native AS (
 SELECT created_at,substring(body from position(chr(10) in body)+1)::json AS doc
 FROM agent_task_updates
 WHERE task_id='a9b2927c-3775-46cc-b86e-2aeb0c2c4479' AND author='hpc-report-a9b2927c'
 AND body LIKE 'HPC3 ATTEMPT LOAD-REPORT%'
), jobs AS (
 SELECT e.*,j.claimed_at,j.closed_at,substring(j.detail from position('{"state"' in j.detail))::jsonb AS outcome
 FROM expected e JOIN fleet_dispatch_jobs j ON j.id::text=e.id
 WHERE j.task_id='a9b2927c-3775-46cc-b86e-2aeb0c2c4479' AND j.command='research' AND j.project='research/hpc3'
 AND j.status='passed' AND j.exit_code=0 AND j.claimed_by='fleet-research-diphtheria'
 AND j.research_request->>'specSha256'=e.spec_sha
 AND j.research_request->>'attempt'='selection-report-9b6cbf367-'||e.number
), verified AS (
 SELECT j.id,j.native_id FROM jobs j
 WHERE j.outcome->>'native_job_id'=j.native_id
 AND j.outcome->>'record_sha256'=j.record_sha
 AND EXISTS (
  SELECT 1 FROM native n
  WHERE n.created_at BETWEEN j.claimed_at AND j.closed_at
  AND n.doc->>'kind'='HPC3_ATTEMPT_EVIDENCE'
  AND n.doc#>>'{ledger,job_id}'=j.native_id
  AND n.doc#>>'{ledger,experiment,research_attempt}'='c28c7c78284a67970c4989dbc1e95048280a71e694722718d4013e25765afe1e:'||j.number
  AND n.doc#>>'{ledger,experiment,request_sha256}'=j.trial_sha
  AND n.doc#>>'{ledger,artifact}'='/data/homezvol1/wagnera3/research/selection-report-9b6cbf367/attempts/c28c7c78284a67970c4989dbc1e95048280a71e694722718d4013e25765afe1e/'||j.number||'/result/evaluation.json'
  AND n.doc#>>'{accounting,job,state}'='COMPLETED'
  AND n.doc#>>'{accounting,exit_code}'='0:0'
  AND n.doc#>>'{measurement,dataset,bundle_sha256}'='e0b1f3b799d1a52ab79c0073e00d9b648ad1b042a70f334405e4a31e69f07b14'
  AND n.doc#>>'{record,experiment}'='3f9724b74004133a6f5d1ad095c5f61ede3c782a0d41d839983122abe923b4ca'
  AND n.doc#>>'{record,fingerprint,image_digest}'='870a9da433e8cd15628724c73f845100e5ddd264969b55364e23c56e04c5d47e'
  AND encode(sha256(convert_to((n.doc->'record')::text,'UTF8')),'hex')=j.record_sha
 )
)
SELECT count(DISTINCT id)||'|'||count(DISTINCT native_id) FROM verified;
