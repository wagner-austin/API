WITH reports AS (
 SELECT id, created_at, substring(body from position(chr(10) in body)+1)::jsonb AS doc
 FROM agent_task_updates
 WHERE task_id='b1934ce6-d021-47f7-b91d-dd4365cf7b7e'
 AND author='hpc-report-b1934ce6'
 AND id IN ('8f3e17b5-9c37-48ef-b1c1-b911ff6b091e','f521619f-ee3a-4ce4-b65d-a2ea724c3bab','485c30c4-3939-4048-8452-757dea4dc0a4','52043d16-9e05-4170-8ca1-6fd2b7c12f35')
), controller AS (
 SELECT * FROM reports WHERE id='8f3e17b5-9c37-48ef-b1c1-b911ff6b091e'
 AND doc->>'kind'='HPC3_CONTROLLER_EVIDENCE'
 AND doc->>'source_commit'='b3843a3d3b83cbae77dc4f855be2c4fbaa17b492'
 AND doc->>'controller_python'='/home/corvis/research-control/wrds-summary-b3843a3d3/bin/python'
 AND doc->>'controller_python_sha256'='ca969be9adfbddc0a593745da84da0348a4c2baf849daae18806a344708934c4'
 AND doc->>'provenance_sha256'='ec1f15f79adbbd3c82518e61a6058c264c2e12041de88def95e93efe2446bb90'
 AND doc->>'selfcheck_sha256'='b388907a6365e694ec6f2f483d598b84920297d79a511fcfb89057116888cfcb'
 AND doc->'verified_wheels'='{"covenant_ml-0.1.0-py3-none-any.whl":{"sha256":"5c43c166ebf927888b8e4e61f70f89549604526b1660e20ebbb3435e738d1572","installed_files":355},"hpc3-0.1.0-py3-none-any.whl":{"sha256":"760ad4644a21a9011a1a739c8ea660f1fd65170192a7f35fab627b02fab162dd","installed_files":134},"platform_core-0.1.0-py3-none-any.whl":{"sha256":"13cd97fdf5a67c6c6882d0029e69fe993fdad0a0fb1c10457c7e64ae47285dea","installed_files":96}}'::jsonb
 AND doc->>'installed_files'='585'
 AND doc->>'selfcheck'='image self-check OK'
 AND doc->>'dependency_check'='No broken requirements found.'
), build_stage AS (
 SELECT * FROM reports WHERE id='f521619f-ee3a-4ce4-b65d-a2ea724c3bab'
 AND doc->>'kind'='HPC3_STAGE_EVIDENCE'
 AND doc->'manifest'='{"destination":"/data/homezvol1/wagnera3/research-images/summary-b3843a3d3","files":[{"name":"build.sbatch","sha256":"91fe5adafbd2b4be33f10b0f2d336934dc00f270d8b0ecb3a355e6966177aee8","size_bytes":2134},{"name":"build.sh","sha256":"11b9eede91eff5d3c80ff0a0635f6b372cdf9eb201b17e3767e6669aed0a3da7","size_bytes":745},{"name":"image.def","sha256":"f37575d20f32f65fd114ffe295a10fb3ce8ebfa3756a462da546c82b391889bc","size_bytes":3124},{"name":"requirements.txt","sha256":"941844d91063e3fdd4d6dacb2a6f52edd44ab71b39342d99a46300d895db47cb","size_bytes":900},{"name":"selfcheck.py","sha256":"b388907a6365e694ec6f2f483d598b84920297d79a511fcfb89057116888cfcb","size_bytes":7325}],"provenance":{"source_commit":"b3843a3d3b83cbae77dc4f855be2c4fbaa17b492","board_task":"b1934ce6-d021-47f7-b91d-dd4365cf7b7e","controller_provenance_sha256":"ec1f15f79adbbd3c82518e61a6058c264c2e12041de88def95e93efe2446bb90","captured_spec_sha256":"30e62360adf9c1310e46609f5bc0181c62785dfc0d29f17b49d04123a498f372"}}'::jsonb
 AND doc->>'verified_files'='5'
), wheels_stage AS (
 SELECT * FROM reports WHERE id='485c30c4-3939-4048-8452-757dea4dc0a4'
 AND doc->>'kind'='HPC3_STAGE_EVIDENCE'
 AND doc->'manifest'='{"destination":"/data/homezvol1/wagnera3/research-images/summary-b3843a3d3/wheels","files":[{"name":"cleargbm-0.1.0-py3-none-any.whl","sha256":"122696d5fd477b908e89c51143af6fd87f476faa58793a53f50001949871b793","size_bytes":34972},{"name":"cleargbm_rs-0.1.0-cp311-cp311-linux_x86_64.whl","sha256":"9541cdcaf10242d974f9f36f652f909a44f8217e1b47920d11258910b00111ad","size_bytes":618460},{"name":"covenant_domain-0.1.0-py3-none-any.whl","sha256":"307dda4c7b6744ec8b50b59645b34073c71621a883de3d879e2ae1386078c0d7","size_bytes":11753},{"name":"covenant_ml-0.1.0-py3-none-any.whl","sha256":"5c43c166ebf927888b8e4e61f70f89549604526b1660e20ebbb3435e738d1572","size_bytes":800320},{"name":"platform_core-0.1.0-py3-none-any.whl","sha256":"13cd97fdf5a67c6c6882d0029e69fe993fdad0a0fb1c10457c7e64ae47285dea","size_bytes":286120},{"name":"platform_ml-0.1.0-py3-none-any.whl","sha256":"bfd11557f154e65eec17c46cbf405ed6dc8f56cf53bb3ea6bcb8e2cbde632501","size_bytes":52844}],"provenance":{"source_commit":"b3843a3d3b83cbae77dc4f855be2c4fbaa17b492","board_task":"b1934ce6-d021-47f7-b91d-dd4365cf7b7e","controller_provenance_sha256":"ec1f15f79adbbd3c82518e61a6058c264c2e12041de88def95e93efe2446bb90","captured_spec_sha256":"30e62360adf9c1310e46609f5bc0181c62785dfc0d29f17b49d04123a498f372"}}'::jsonb
 AND doc->>'verified_files'='6'
), image AS (
 SELECT * FROM reports WHERE id='52043d16-9e05-4170-8ca1-6fd2b7c12f35'
 AND doc->>'kind'='HPC3_IMAGE_BUILD_EVIDENCE'
 AND doc#>>'{accounting,job,job_id}'='58719937'
 AND doc#>>'{accounting,job,state}'='COMPLETED'
 AND doc#>>'{accounting,job,partition}'='free'
 AND doc#>>'{accounting,exit_code}'='0:0'
 AND doc->>'selfcheck_passes'='2'
 AND doc->>'dependency_check'='passed'
 AND doc->'verified_files'='{"/data/homezvol1/wagnera3/research-images/summary-b3843a3d3/build.sbatch":"91fe5adafbd2b4be33f10b0f2d336934dc00f270d8b0ecb3a355e6966177aee8","/data/homezvol1/wagnera3/research-images/summary-b3843a3d3/build.sh":"11b9eede91eff5d3c80ff0a0635f6b372cdf9eb201b17e3767e6669aed0a3da7","/data/homezvol1/wagnera3/research-images/summary-b3843a3d3/image.def":"f37575d20f32f65fd114ffe295a10fb3ce8ebfa3756a462da546c82b391889bc","/data/homezvol1/wagnera3/research-images/summary-b3843a3d3/requirements.txt":"941844d91063e3fdd4d6dacb2a6f52edd44ab71b39342d99a46300d895db47cb","/data/homezvol1/wagnera3/research-images/summary-b3843a3d3/selfcheck.py":"b388907a6365e694ec6f2f483d598b84920297d79a511fcfb89057116888cfcb","/data/homezvol1/wagnera3/research-images/summary-b3843a3d3/wheels/cleargbm-0.1.0-py3-none-any.whl":"122696d5fd477b908e89c51143af6fd87f476faa58793a53f50001949871b793","/data/homezvol1/wagnera3/research-images/summary-b3843a3d3/wheels/cleargbm_rs-0.1.0-cp311-cp311-linux_x86_64.whl":"9541cdcaf10242d974f9f36f652f909a44f8217e1b47920d11258910b00111ad","/data/homezvol1/wagnera3/research-images/summary-b3843a3d3/wheels/covenant_domain-0.1.0-py3-none-any.whl":"307dda4c7b6744ec8b50b59645b34073c71621a883de3d879e2ae1386078c0d7","/data/homezvol1/wagnera3/research-images/summary-b3843a3d3/wheels/covenant_ml-0.1.0-py3-none-any.whl":"5c43c166ebf927888b8e4e61f70f89549604526b1660e20ebbb3435e738d1572","/data/homezvol1/wagnera3/research-images/summary-b3843a3d3/wheels/platform_core-0.1.0-py3-none-any.whl":"13cd97fdf5a67c6c6882d0029e69fe993fdad0a0fb1c10457c7e64ae47285dea","/data/homezvol1/wagnera3/research-images/summary-b3843a3d3/wheels/platform_ml-0.1.0-py3-none-any.whl":"bfd11557f154e65eec17c46cbf405ed6dc8f56cf53bb3ea6bcb8e2cbde632501","/data/homezvol1/wagnera3/research-images/summary-b3843a3d3/GIT_COMMIT":"243dae65b81e7aa10dfb8f5d25c7a64f16adb50c431de8f0468ccfaee787afe4","/data/homezvol1/wagnera3/research-images/summary-b3843a3d3/research-summary.sif":"14b1b5c929b3e89f62edae19cb04714e66c1d122273296e6d532559d717f8f4e","/data/homezvol1/wagnera3/research-images/summary-b3843a3d3/build-58719937.out":"169b40c13ca44c15696f03c5a9d317145ecf94f7e36c6e5c2f0973875ffe5cea"}'::jsonb
)
SELECT count(*) FROM controller, build_stage, wheels_stage, image
WHERE controller.created_at < build_stage.created_at
AND build_stage.created_at < wheels_stage.created_at
AND wheels_stage.created_at < image.created_at;
