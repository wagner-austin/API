# Protocol-v2 controller and HPC3 image installation

Board task: `1ec0dbcb-4c25-455f-9c16-b2b8e3177a68`.
Source: API `8ef7e0f1769e7127feabb30008d72bc70e908111`.

`evidence.json` binds the controller inspection, native image accounting,
service posts and SHA-256 identities of the retained metadata. The controller
inspection verified 623 installed members against seven frozen first-party
wheels. Native image job `58739821` completed in 88 seconds with exit `0:0`;
inspection verified fourteen pinned files and two symbol self-check passes.
Both dependency checks passed.

Construction reused `fleet.core.export.archive_commit`, Loki wheel builds,
`hpc3.cli.image_capture`, image rendering, staging, native image building,
watching and inspection. The unchanged compiled ClearGBM wheel was checked
against its historical frozen digest and unchanged source. Shared library
wheels were installed together into a fresh controller environment.

`capture-config.json` identifies the historical image used to capture the
53 dependency pins. It is capture provenance, not the new study submission
configuration. The new installed image identity is in `image-report.json`.

This is installation evidence only. No current-runtime queue rehearsal,
licensed evaluation, candidate selection or final grant release occurred.
Old controllers, images, ciphertext and journals remain historical records.
These files contain public runtime metadata; they contain no private keys or
licensed observations. New study resources and the complete synthetic queue
rehearsal must be recorded separately before the substantive study resumes.
