SELECT count(*)
FROM fleet_dispatch_jobs AS job
WHERE job.id = '872b9e09-d808-414a-aa3a-c93b9f8a8f3f'
  AND job.status = 'passed'
  AND job.exit_code = 0
  AND job.detail LIKE '%"native_job_id":"40554174"%'
  AND job.detail LIKE '%b93adc4e218c707fb3cf883df598e03656dcf400c5274433392b1a8d236d5bde%'
  AND EXISTS (
    SELECT 1 FROM agent_task_updates AS native
    WHERE native.task_id = 'a622165c-bb63-4714-933b-10d527f21478'
      AND native.author = 'load-report-a622165c'
      AND native.created_at >= '2026-10-31T19:11:00Z'
      AND native.body LIKE 'WRDS ATTEMPT LOAD-REPORT %job 40554174 on wrds%'
      AND native.body LIKE '%identity 51da76ff3aa3a0c547f4b6627b1fe2475a2d473982c183546485fff1b6e0a79c%'
      AND native.body LIKE '%accounting job 40554174 hostname=wrds-sas31-w.wharton.private failed=0 exit_status=0%'
      AND native.body LIKE '%output extraction.json sha256=18bd2028207c56784df784916728b62800b7bc3c6a89ea5b1f5970c7c28f9349%'
      AND native.body LIKE '%output train-validation/bundle.json sha256=26b46d56c5ab47565813981f0eb4dabad4e0d5260c86dd02cf45becbe6c858f1%'
      AND native.body LIKE '%output final/bundle.json sha256=377b316ee84d185853412776b63ec8ccb6da7ecd0cbe4d6293bfa703ca0c226e%'
      AND native.body LIKE '%run-record sha256=b93adc4e218c707fb3cf883df598e03656dcf400c5274433392b1a8d236d5bde%'
      AND (SELECT count(*) FROM unnest(string_to_array(native.body, chr(10))) AS line
           WHERE line LIKE 'output % sha256=%') = 24
      AND (SELECT count(*) FROM unnest(string_to_array(native.body, chr(10))) AS line
           WHERE line LIKE 'output %.age sha256=%') = 20
  )
  AND EXISTS (
    SELECT 1 FROM agent_task_updates AS installed
    WHERE installed.task_id = 'da2a55dc-dc5a-4f4b-9bb7-81fbcf75f320'
      AND installed.author = 'hpc-report-da2a55dc'
      AND installed.body LIKE 'HPC3 CONTROLLER LOAD-REPORT%'
      AND installed.body LIKE '%"controller_python":"/home/corvis/research-control/wrds-queue-0ffb98731/bin/python"%'
      AND installed.body LIKE '%"fleet-0.1.0-py3-none-any.whl":{"sha256":"384056b601e5792c9c543b6ebc7ef5c0f6bd06a8216cd3adfdc6400a65f9e37f","installed_files":185}%'
      AND installed.body LIKE '%"installed_files":872%'
      AND installed.body LIKE '%"dependency_check":"No broken requirements found."%'
  );
