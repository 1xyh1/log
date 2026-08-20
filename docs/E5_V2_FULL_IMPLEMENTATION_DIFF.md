# E5 v2 完整实现差异

本文档记录从收到的 E5 failure bundle 到 E5 v2 的完整代码/配置差异。所有修改块均为 unified diff；新增、删除和上下文行完整保留。运行证据和原始失败日志不属于实现代码，未纳入本差异。

```diff
diff -ruN /workspace/scratch/2d09435bab69/tmp/review/e5_failure/config/t1gr_e5_security_policy.json /workspace/scratch/2d09435bab69/work/e5_v2/config/t1gr_e5_security_policy.json
--- /workspace/scratch/2d09435bab69/tmp/review/e5_failure/config/t1gr_e5_security_policy.json	2026-08-20 02:55:26.000000000 +0800
+++ /workspace/scratch/2d09435bab69/work/e5_v2/config/t1gr_e5_security_policy.json	2026-08-20 13:13:41.801226886 +0800
@@ -1,5 +1,5 @@
 {
-  "schema": "t1gr-e5-security-policy-v1",
+  "schema": "t1gr-e5-security-policy-v2",
   "public_output_prefix": "reports/step4_t1gr",
   "private_parent_must_preexist": true,
   "max_private_json_bytes": 67108864,
@@ -19,5 +19,7 @@
   "ultralytics_usage_analytics": "DISABLED_FOR_E5_PROCESS",
   "amp_network_probe": "BYPASSED; AMP is qualified by mandatory 1-epoch smoke before formal train",
   "posix_private_artifact_umask": "0077",
-  "windows_private_acl_claim": "NOT_PROVEN_BY_PYTHON; private roots must be user-controlled"
-}
\ No newline at end of file
+  "windows_private_acl_claim": "NOT_PROVEN_BY_PYTHON; private roots must be user-controlled",
+  "private_failure_traceback": "PRIVATE_RUN_DIRECTORY_ONLY; NEVER COPIED TO PUBLIC REPORTS",
+  "private_failure_traceback_max_bytes": 1048576
+}
diff -ruN /workspace/scratch/2d09435bab69/tmp/review/e5_failure/config/t1gr_e5_training_spec.candidate.json /workspace/scratch/2d09435bab69/work/e5_v2/config/t1gr_e5_training_spec.candidate.json
--- /workspace/scratch/2d09435bab69/tmp/review/e5_failure/config/t1gr_e5_training_spec.candidate.json	2026-08-20 02:55:26.000000000 +0800
+++ /workspace/scratch/2d09435bab69/work/e5_v2/config/t1gr_e5_training_spec.candidate.json	2026-08-20 13:02:50.377234221 +0800
@@ -1,7 +1,7 @@
 {
-  "schema": "t1gr-e5-training-spec-v1",
+  "schema": "t1gr-e5-training-spec-v2",
   "status": "CANDIDATE_REQUIRES_REVIEW",
-  "notes": "Known project choices: epochs=80,batch=4,imgsz=640,seed=20260812,end2end=true. Other values are explicit Ultralytics v8.4.56 defaults, except optimizer is resolved from auto to explicit MuSGD for >10000 iterations and warmup_bias_lr=0.0 to match that auto branch. Review workers/device before setting status=REVIEWED_FROZEN.",
+  "notes": "Candidate copy of the E5 v2 adjudicated recipe. Set status=REVIEWED_FROZEN_V2 only after review; MuSGD is a deliberate project choice and is not auto-equivalent.",
   "architecture": "yolo26s",
   "model_yaml": "yolo26s.yaml",
   "num_classes": 12,
@@ -63,7 +63,7 @@
     "split": "val",
     "conf": 0.001,
     "iou": 0.7,
-    "max_det": 300,
+    "max_det": 100,
     "half": false,
     "dnn": false,
     "plots": false,
@@ -76,6 +76,28 @@
     "formal_timeout_seconds": 43200,
     "eval_timeout_seconds": 3600,
     "lock_wait_seconds": 5.0,
-    "lock_stale_seconds": 900.0
+    "lock_stale_seconds": 900.0,
+    "private_traceback_max_bytes": 1048576
+  },
+  "review_freeze": {
+    "date": "2026-08-20",
+    "basis": "project-original E5 recipe retained after v2 optimizer adjudication",
+    "workers_device_policy": "workers=8 and device=0 are frozen; no CLI override",
+    "authority": "E5 Step1 mother baseline; subsequent matched comparisons inherit all non-treatment settings",
+    "optimizer_adjudication": {
+      "decision": "KEEP_PROJECT_FROZEN_MUSGD",
+      "selected_optimizer": "MuSGD",
+      "framework_auto_would_select": "AdamW",
+      "train_sample_count": 1504,
+      "nominal_batch": 4,
+      "nbs": 64,
+      "epochs": 80,
+      "framework_iteration_formula": "ceil(train_sample_count/max(nominal_batch,nbs))*epochs",
+      "framework_estimated_iterations": 1920,
+      "auto_threshold_iterations": 10000,
+      "not_auto_equivalent": true,
+      "rationale": "Preserve the already frozen mother-baseline optimizer and downstream comparison continuity. The scientific choice is explicit MuSGD, not runtime auto-selection."
+    },
+    "evaluation_detection_cap": 100
   }
-}
\ No newline at end of file
+}
diff -ruN /workspace/scratch/2d09435bab69/tmp/review/e5_failure/config/t1gr_e5_training_spec.frozen.json /workspace/scratch/2d09435bab69/work/e5_v2/config/t1gr_e5_training_spec.frozen.json
--- /workspace/scratch/2d09435bab69/tmp/review/e5_failure/config/t1gr_e5_training_spec.frozen.json	2026-08-20 02:55:26.000000000 +0800
+++ /workspace/scratch/2d09435bab69/work/e5_v2/config/t1gr_e5_training_spec.frozen.json	2026-08-20 13:02:50.305234221 +0800
@@ -1,7 +1,7 @@
 {
-  "schema": "t1gr-e5-training-spec-v1",
-  "status": "REVIEWED_FROZEN",
-  "notes": "Formal E5 Step1 RGB mother-baseline freeze. Project-known values: epochs=80, batch=4, imgsz=640, seed=20260812, YOLO26s end2end=true. Remaining training/eval values are explicitly frozen from Ultralytics v8.4.56 defaults, except optimizer=auto is resolved before execution to the source-equivalent MuSGD branch for the estimated >10000-iteration regime (lr0=0.01, momentum=0.9, warmup_bias_lr=0.0). workers=8 and device=0 are explicitly frozen; no scientific CLI override is permitted.",
+  "schema": "t1gr-e5-training-spec-v2",
+  "status": "REVIEWED_FROZEN_V2",
+  "notes": "E5 v2 Step1 RGB mother-baseline freeze. The project-original explicit MuSGD recipe is retained by adjudication for baseline continuity. This is not represented as equivalent to Ultralytics optimizer=auto: with 1504 training samples, nbs=64 and 80 epochs, the v8.4.56 auto-optimizer iteration estimate is ceil(1504/64)*80=1920 and would select AdamW. Scientific CLI overrides remain forbidden. Evaluation max_det is fixed to the official per-image cap 100.",
   "architecture": "yolo26s",
   "model_yaml": "yolo26s.yaml",
   "num_classes": 12,
@@ -63,7 +63,7 @@
     "split": "val",
     "conf": 0.001,
     "iou": 0.7,
-    "max_det": 300,
+    "max_det": 100,
     "half": false,
     "dnn": false,
     "plots": false,
@@ -76,14 +76,28 @@
     "formal_timeout_seconds": 43200,
     "eval_timeout_seconds": 3600,
     "lock_wait_seconds": 5.0,
-    "lock_stale_seconds": 900.0
+    "lock_stale_seconds": 900.0,
+    "private_traceback_max_bytes": 1048576
   },
   "review_freeze": {
     "date": "2026-08-20",
-    "basis": "project-known 80ep/batch4/imgsz640/seed20260812 + explicit Ultralytics v8.4.56 defaults; optimizer auto resolved to MuSGD branch",
-    "workers_device_policy": "workers=8 and device=0 frozen as explicit runtime values; no CLI override",
-    "authority": "E5 Step1 mother baseline recipe; G0/G1/G2 must inherit matched non-treatment settings",
-    "estimated_training_iterations": 30080,
-    "optimizer_resolution": "estimated iterations > 10000; Ultralytics v8.4.56 auto branch => MuSGD lr0=0.01 momentum=0.9 and warmup_bias_lr=0.0; frozen explicitly to remove runtime auto-selection"
+    "basis": "project-original E5 recipe retained after v2 optimizer adjudication",
+    "workers_device_policy": "workers=8 and device=0 are frozen; no CLI override",
+    "authority": "E5 Step1 mother baseline; subsequent matched comparisons inherit all non-treatment settings",
+    "optimizer_adjudication": {
+      "decision": "KEEP_PROJECT_FROZEN_MUSGD",
+      "selected_optimizer": "MuSGD",
+      "framework_auto_would_select": "AdamW",
+      "train_sample_count": 1504,
+      "nominal_batch": 4,
+      "nbs": 64,
+      "epochs": 80,
+      "framework_iteration_formula": "ceil(train_sample_count/max(nominal_batch,nbs))*epochs",
+      "framework_estimated_iterations": 1920,
+      "auto_threshold_iterations": 10000,
+      "not_auto_equivalent": true,
+      "rationale": "Preserve the already frozen mother-baseline optimizer and downstream comparison continuity. The scientific choice is explicit MuSGD, not runtime auto-selection."
+    },
+    "evaluation_detection_cap": 100
   }
-}
\ No newline at end of file
+}
diff -ruN /workspace/scratch/2d09435bab69/tmp/review/e5_failure/config/t1gr_e5_training_spec.template.json /workspace/scratch/2d09435bab69/work/e5_v2/config/t1gr_e5_training_spec.template.json
--- /workspace/scratch/2d09435bab69/tmp/review/e5_failure/config/t1gr_e5_training_spec.template.json	2026-08-20 02:55:26.000000000 +0800
+++ /workspace/scratch/2d09435bab69/work/e5_v2/config/t1gr_e5_training_spec.template.json	2026-08-20 13:02:50.445234220 +0800
@@ -1,7 +1,7 @@
 {
-  "schema": "t1gr-e5-training-spec-v1",
-  "status": "TEMPLATE_UNRESOLVED",
-  "notes": "Fill every null and review explicitly. Formal recipe builder accepts only status=REVIEWED_FROZEN.",
+  "schema": "t1gr-e5-training-spec-v2",
+  "status": "TEMPLATE_ONLY",
+  "notes": "Template copy. Review all fields and the explicit optimizer adjudication before freezing.",
   "architecture": "yolo26s",
   "model_yaml": "yolo26s.yaml",
   "num_classes": 12,
@@ -26,7 +26,7 @@
     "dfl": 1.5,
     "cos_lr": false,
     "amp": true,
-    "workers": null,
+    "workers": 8,
     "deterministic": true,
     "cache": false,
     "rect": false,
@@ -63,19 +63,41 @@
     "split": "val",
     "conf": 0.001,
     "iou": 0.7,
-    "max_det": 300,
+    "max_det": 100,
     "half": false,
     "dnn": false,
     "plots": false,
     "save_json": false
   },
   "runtime": {
-    "device": null,
+    "device": "0",
     "smoke_epochs": 1,
     "smoke_timeout_seconds": 1800,
     "formal_timeout_seconds": 43200,
     "eval_timeout_seconds": 3600,
     "lock_wait_seconds": 5.0,
-    "lock_stale_seconds": 900.0
+    "lock_stale_seconds": 900.0,
+    "private_traceback_max_bytes": 1048576
+  },
+  "review_freeze": {
+    "date": "2026-08-20",
+    "basis": "project-original E5 recipe retained after v2 optimizer adjudication",
+    "workers_device_policy": "workers=8 and device=0 are frozen; no CLI override",
+    "authority": "E5 Step1 mother baseline; subsequent matched comparisons inherit all non-treatment settings",
+    "optimizer_adjudication": {
+      "decision": "KEEP_PROJECT_FROZEN_MUSGD",
+      "selected_optimizer": "MuSGD",
+      "framework_auto_would_select": "AdamW",
+      "train_sample_count": 1504,
+      "nominal_batch": 4,
+      "nbs": 64,
+      "epochs": 80,
+      "framework_iteration_formula": "ceil(train_sample_count/max(nominal_batch,nbs))*epochs",
+      "framework_estimated_iterations": 1920,
+      "auto_threshold_iterations": 10000,
+      "not_auto_equivalent": true,
+      "rationale": "Preserve the already frozen mother-baseline optimizer and downstream comparison continuity. The scientific choice is explicit MuSGD, not runtime auto-selection."
+    },
+    "evaluation_detection_cap": 100
   }
-}
\ No newline at end of file
+}

Binary files /workspace/scratch/2d09435bab69/tmp/review/e5_failure/scripts/__pycache__/t1gr_e5_build_rgb_view.cpython-312.pyc and /workspace/scratch/2d09435bab69/work/e5_v2/scripts/__pycache__/t1gr_e5_build_rgb_view.cpython-312.pyc differ
Binary files /workspace/scratch/2d09435bab69/tmp/review/e5_failure/scripts/__pycache__/t1gr_e5_eval_step1.cpython-312.pyc and /workspace/scratch/2d09435bab69/work/e5_v2/scripts/__pycache__/t1gr_e5_eval_step1.cpython-312.pyc differ
Binary files /workspace/scratch/2d09435bab69/tmp/review/e5_failure/scripts/__pycache__/t1gr_e5_final_audit.cpython-312.pyc and /workspace/scratch/2d09435bab69/work/e5_v2/scripts/__pycache__/t1gr_e5_final_audit.cpython-312.pyc differ
Binary files /workspace/scratch/2d09435bab69/tmp/review/e5_failure/scripts/__pycache__/t1gr_e5_freeze_recipe.cpython-312.pyc and /workspace/scratch/2d09435bab69/work/e5_v2/scripts/__pycache__/t1gr_e5_freeze_recipe.cpython-312.pyc differ
Binary files /workspace/scratch/2d09435bab69/tmp/review/e5_failure/scripts/__pycache__/t1gr_e5_preflight.cpython-312.pyc and /workspace/scratch/2d09435bab69/work/e5_v2/scripts/__pycache__/t1gr_e5_preflight.cpython-312.pyc differ
Binary files /workspace/scratch/2d09435bab69/tmp/review/e5_failure/scripts/__pycache__/t1gr_e5_run_step1.cpython-312.pyc and /workspace/scratch/2d09435bab69/work/e5_v2/scripts/__pycache__/t1gr_e5_run_step1.cpython-312.pyc differ
Binary files /workspace/scratch/2d09435bab69/tmp/review/e5_failure/scripts/__pycache__/t1gr_e5_v2_regression_gate.cpython-312.pyc and /workspace/scratch/2d09435bab69/work/e5_v2/scripts/__pycache__/t1gr_e5_v2_regression_gate.cpython-312.pyc differ
diff -ruN /workspace/scratch/2d09435bab69/tmp/review/e5_failure/scripts/t1gr_e5_build_rgb_view.py /workspace/scratch/2d09435bab69/work/e5_v2/scripts/t1gr_e5_build_rgb_view.py
--- /workspace/scratch/2d09435bab69/tmp/review/e5_failure/scripts/t1gr_e5_build_rgb_view.py	2026-08-20 02:58:18.000000000 +0800
+++ /workspace/scratch/2d09435bab69/work/e5_v2/scripts/t1gr_e5_build_rgb_view.py	2026-08-20 13:13:42.065226883 +0800
@@ -18,7 +18,7 @@
     scan_formal_zip, validate_e4_evidence, verify_view_tree,
 )
 
-SCRIPT_VERSION="t1gr-e5-build-rgb-view-hardened-v1"
+SCRIPT_VERSION="t1gr-e5-v2-build-rgb-view-hardened-v2"
 
 def raw_sha(b:bytes)->str: return hashlib.sha256(b).hexdigest()
 
@@ -36,8 +36,8 @@
     repo=ROOT.resolve(strict=True)
     secp=ensure_repo_input(repo,"config/t1gr_e5_security_policy.json","config")
     if sha256_file(secp)!=FROZEN_E5_SECURITY_POLICY_SHA256: fail("E5_SECURITY_POLICY_SHA_DRIFT")
-    sec=read_json_bounded(secp,1<<20,"t1gr-e5-security-policy-v1")
-    rp=ensure_repo_input(repo,"reports/step4_t1gr/e5_step1_recipe_public.json","reports/step4_t1gr")
+    sec=read_json_bounded(secp,1<<20,"t1gr-e5-security-policy-v2")
+    rp=ensure_repo_input(repo,"reports/step4_t1gr/e5_v2_step1_recipe_public.json","reports/step4_t1gr")
     fp=ensure_repo_input(repo,"reports/step4_t1gr/e4_split_freeze_public.json","reports/step4_t1gr")
     vp=ensure_repo_input(repo,"reports/step4_t1gr/e4_seal_verification_public.json","reports/step4_t1gr")
     td_p=ensure_private_input(Path(a.train_dev_access),repo)
@@ -46,7 +46,7 @@
     if not out_root.parent.is_dir(): fail("E5_VIEW_PARENT_NOT_FOUND")
     if not os.access(out_root.parent,os.W_OK): fail("E5_VIEW_PARENT_NOT_WRITABLE")
     if out_root.exists() and not out_root.is_dir(): fail("E5_VIEW_ROOT_NOT_DIRECTORY")
-    pub=ensure_public_output(repo,"reports/step4_t1gr/e5_step1_view_public.json",sec["public_output_prefix"])
+    pub=ensure_public_output(repo,"reports/step4_t1gr/e5_v2_step1_view_public.json",sec["public_output_prefix"])
     zp=Path(a.formal_zip).expanduser().resolve(strict=False)
     if not zp.is_file():fail("FORMAL_ZIP_NOT_FOUND")
     deadline=Deadline(float(a.timeout_seconds or sec["view_build_timeout_seconds"]))
diff -ruN /workspace/scratch/2d09435bab69/tmp/review/e5_failure/scripts/t1gr_e5_eval_step1.py /workspace/scratch/2d09435bab69/work/e5_v2/scripts/t1gr_e5_eval_step1.py
--- /workspace/scratch/2d09435bab69/tmp/review/e5_failure/scripts/t1gr_e5_eval_step1.py	2026-08-20 02:55:26.000000000 +0800
+++ /workspace/scratch/2d09435bab69/work/e5_v2/scripts/t1gr_e5_eval_step1.py	2026-08-20 13:13:42.329226880 +0800
@@ -15,7 +15,7 @@
  environment_probe,payload_ok,private_umask,ultralytics_offline_guard,verify_view_tree,wall_clock_watchdog
 )
 
-SCRIPT_VERSION="t1gr-e5-step1-dev-eval-hardened-v1"
+SCRIPT_VERSION="t1gr-e5-v2-step1-dev-eval-hardened-v2"
 
 def private_run_root(path:Path,repo:Path)->Path:
     p=path.expanduser().resolve(strict=False)
@@ -27,12 +27,12 @@
     repo=ROOT.resolve(strict=True)
     secp=ensure_repo_input(repo,"config/t1gr_e5_security_policy.json","config")
     if sha256_file(secp)!=FROZEN_E5_SECURITY_POLICY_SHA256: fail("E5_SECURITY_POLICY_SHA_DRIFT")
-    sec=read_json_bounded(secp,1<<20,"t1gr-e5-security-policy-v1")
-    rp=ensure_repo_input(repo,"reports/step4_t1gr/e5_step1_recipe_public.json","reports/step4_t1gr")
-    vpubp=ensure_repo_input(repo,"reports/step4_t1gr/e5_step1_view_public.json","reports/step4_t1gr")
-    runp=ensure_repo_input(repo,"reports/step4_t1gr/e5_step1_formal_run_public.json","reports/step4_t1gr")
+    sec=read_json_bounded(secp,1<<20,"t1gr-e5-security-policy-v2")
+    rp=ensure_repo_input(repo,"reports/step4_t1gr/e5_v2_step1_recipe_public.json","reports/step4_t1gr")
+    vpubp=ensure_repo_input(repo,"reports/step4_t1gr/e5_v2_step1_view_public.json","reports/step4_t1gr")
+    runp=ensure_repo_input(repo,"reports/step4_t1gr/e5_v2_step1_formal_run_public.json","reports/step4_t1gr")
     td_p=ensure_private_input(Path(a.train_dev_access),repo);vm_p=ensure_private_input(Path(a.view_manifest),repo)
-    out=ensure_public_output(repo,"reports/step4_t1gr/e5_step1_eval_public.json",sec["public_output_prefix"])
+    out=ensure_public_output(repo,"reports/step4_t1gr/e5_v2_step1_eval_public.json",sec["public_output_prefix"])
     recipe=read_json_bounded(rp,int(sec["max_public_json_bytes"]),SCHEMA_RECIPE)
     vpub=read_json_bounded(vpubp,int(sec["max_public_json_bytes"]),SCHEMA_VIEW_PUBLIC)
     rr=read_json_bounded(runp,int(sec["max_public_json_bytes"]),SCHEMA_RUN)
@@ -42,7 +42,7 @@
     if rr.get("mode")!="formal" or rr.get("run_gate_passed") is not True or rr.get("dev_eval_authorized") is not True:
         fail("E5_FORMAL_RUN_NOT_EVALUABLE")
     run_root=private_run_root(Path(a.run_root),repo)
-    run_dir=run_root/"STEP1_RGB_BASELINE";last=run_dir/"weights"/"last.pt"
+    run_dir=run_root/"STEP1_RGB_BASELINE_V2";last=run_dir/"weights"/"last.pt"
     if not last.is_file():fail("E5_FORMAL_LAST_PT_MISSING")
     deadline=Deadline(float(recipe["runtime"]["eval_timeout_seconds"]))
     with file_lock(out.with_suffix(out.suffix+".lock"),float(recipe["runtime"]["lock_wait_seconds"]),float(recipe["runtime"]["lock_stale_seconds"])):
@@ -66,7 +66,7 @@
             from ultralytics import YOLO
         except Exception:fail("E5_EVAL_IMPORT_FAIL")
         if str(ultralytics.__version__)!=recipe["environment"]["ultralytics_version"]:fail("E5_EVAL_ULTRALYTICS_DRIFT")
-        eval_dir=run_root/"STEP1_RGB_DEV_EVAL"
+        eval_dir=run_root/"STEP1_RGB_DEV_EVAL_V2"
         if eval_dir.exists(): fail("E5_EVAL_DIRECTORY_ALREADY_EXISTS")
         offline_state={};permission_state={}
         with ultralytics_offline_guard(bypass_amp_download_check=False) as og, private_umask() as pg:
@@ -88,7 +88,7 @@
             offline_state.update(og);permission_state.update(pg)
             with wall_clock_watchdog(float(recipe["runtime"]["eval_timeout_seconds"]),"E5_EVAL_TIMEOUT"):
                 result=y.val(data=str(vr["dataset_yaml"]),split="val",device=recipe["runtime"]["device"],
-                             project=str(run_root),name="STEP1_RGB_DEV_EVAL",exist_ok=False,verbose=False,**ea)
+                             project=str(run_root),name="STEP1_RGB_DEV_EVAL_V2",exist_ok=False,verbose=False,**ea)
         box=getattr(result,"box",None)
         if box is None:fail("E5_EVAL_BOX_METRICS_MISSING")
         maps=[float(x) for x in getattr(box,"maps",[])]
diff -ruN /workspace/scratch/2d09435bab69/tmp/review/e5_failure/scripts/t1gr_e5_final_audit.py /workspace/scratch/2d09435bab69/work/e5_v2/scripts/t1gr_e5_final_audit.py
--- /workspace/scratch/2d09435bab69/tmp/review/e5_failure/scripts/t1gr_e5_final_audit.py	2026-08-20 02:55:26.000000000 +0800
+++ /workspace/scratch/2d09435bab69/work/e5_v2/scripts/t1gr_e5_final_audit.py	2026-08-20 13:13:42.425226879 +0800
@@ -12,24 +12,24 @@
  FROZEN_E5_SECURITY_POLICY_SHA256,FROZEN_E5_TRAINING_SPEC_SHA256,
  SCHEMA_EVAL,SCHEMA_FINAL,SCHEMA_PREFLIGHT,SCHEMA_RECIPE,SCHEMA_RUN,SCHEMA_VIEW_PUBLIC,payload_ok,parse_utc
 )
-SCRIPT_VERSION="t1gr-e5-final-audit-hardened-v1"
+SCRIPT_VERSION="t1gr-e5-v2-final-audit-hardened-v2"
 
 def run(a):
  repo=ROOT.resolve(strict=True)
  secp=ensure_repo_input(repo,"config/t1gr_e5_security_policy.json","config")
  if sha256_file(secp)!=FROZEN_E5_SECURITY_POLICY_SHA256: fail("E5_SECURITY_POLICY_SHA_DRIFT")
- sec=read_json_bounded(secp,1<<20,"t1gr-e5-security-policy-v1")
+ sec=read_json_bounded(secp,1<<20,"t1gr-e5-security-policy-v2")
  paths={
   "e4_freeze":ensure_repo_input(repo,"reports/step4_t1gr/e4_split_freeze_public.json","reports/step4_t1gr"),
   "e4_verify":ensure_repo_input(repo,"reports/step4_t1gr/e4_seal_verification_public.json","reports/step4_t1gr"),
-  "recipe":ensure_repo_input(repo,"reports/step4_t1gr/e5_step1_recipe_public.json","reports/step4_t1gr"),
-  "view":ensure_repo_input(repo,"reports/step4_t1gr/e5_step1_view_public.json","reports/step4_t1gr"),
-  "preflight":ensure_repo_input(repo,"reports/step4_t1gr/e5_step1_preflight_public.json","reports/step4_t1gr"),
-  "smoke":ensure_repo_input(repo,"reports/step4_t1gr/e5_step1_smoke_public.json","reports/step4_t1gr"),
-  "formal":ensure_repo_input(repo,"reports/step4_t1gr/e5_step1_formal_run_public.json","reports/step4_t1gr"),
-  "eval":ensure_repo_input(repo,"reports/step4_t1gr/e5_step1_eval_public.json","reports/step4_t1gr"),
+  "recipe":ensure_repo_input(repo,"reports/step4_t1gr/e5_v2_step1_recipe_public.json","reports/step4_t1gr"),
+  "view":ensure_repo_input(repo,"reports/step4_t1gr/e5_v2_step1_view_public.json","reports/step4_t1gr"),
+  "preflight":ensure_repo_input(repo,"reports/step4_t1gr/e5_v2_step1_preflight_public.json","reports/step4_t1gr"),
+  "smoke":ensure_repo_input(repo,"reports/step4_t1gr/e5_v2_step1_smoke_public.json","reports/step4_t1gr"),
+  "formal":ensure_repo_input(repo,"reports/step4_t1gr/e5_v2_step1_formal_run_public.json","reports/step4_t1gr"),
+  "eval":ensure_repo_input(repo,"reports/step4_t1gr/e5_v2_step1_eval_public.json","reports/step4_t1gr"),
  }
- out=ensure_public_output(repo,"reports/step4_t1gr/e5_final_audit_public.json",sec["public_output_prefix"])
+ out=ensure_public_output(repo,"reports/step4_t1gr/e5_v2_final_audit_public.json",sec["public_output_prefix"])
  with file_lock(out.with_suffix(out.suffix+".lock"),5.0,900.0):
   obj={k:read_json_bounded(p,int(sec["max_public_json_bytes"])) for k,p in paths.items()}
   if obj["recipe"].get("schema")!=SCHEMA_RECIPE or obj["view"].get("schema")!=SCHEMA_VIEW_PUBLIC or obj["preflight"].get("schema")!=SCHEMA_PREFLIGHT:
@@ -58,6 +58,12 @@
    "formal_view_pin":obj["formal"].get("view_public_sha256")==view_sha,
    "formal_preflight_pin":obj["formal"].get("preflight_public_sha256")==pre_sha,
    "formal_smoke_pin":obj["formal"].get("smoke_report_sha256")==smoke_sha,
+   "initial_state_preflight_smoke_pin":isinstance(obj["preflight"].get("model_preflight",{}).get("model_initial_state_sha256"),str) and obj["preflight"].get("model_preflight",{}).get("model_initial_state_sha256")==obj["smoke"].get("model_initial_state_sha256"),
+   "initial_state_preflight_formal_pin":isinstance(obj["preflight"].get("model_preflight",{}).get("model_initial_state_sha256"),str) and obj["preflight"].get("model_preflight",{}).get("model_initial_state_sha256")==obj["formal"].get("model_initial_state_sha256"),
+   "smoke_training_start_state_pin":obj["smoke"].get("model_initial_state_sha256")==obj["smoke"].get("training_start_state_sha256"),
+   "formal_training_start_state_pin":obj["formal"].get("model_initial_state_sha256")==obj["formal"].get("training_start_state_sha256"),
+   "untransferred_state_preflight_smoke_pin":isinstance(obj["preflight"].get("model_preflight",{}).get("untransferred_initial_state_sha256"),str) and obj["preflight"].get("model_preflight",{}).get("untransferred_initial_state_sha256")==obj["smoke"].get("untransferred_initial_state_sha256"),
+   "untransferred_state_preflight_formal_pin":isinstance(obj["preflight"].get("model_preflight",{}).get("untransferred_initial_state_sha256"),str) and obj["preflight"].get("model_preflight",{}).get("untransferred_initial_state_sha256")==obj["formal"].get("untransferred_initial_state_sha256"),
    "formal_exact_epochs":obj["formal"].get("epochs_completed")==obj["formal"].get("epochs_expected")==obj["recipe"]["train_args"]["epochs"],
    "eval_pass":obj["eval"].get("eval_gate_passed") is True and obj["eval"].get("authority")=="DEV_ONLY_DIAGNOSTIC_BASELINE",
    "eval_recipe_pin":obj["eval"].get("recipe_public_sha256")==recipe_sha,
diff -ruN /workspace/scratch/2d09435bab69/tmp/review/e5_failure/scripts/t1gr_e5_freeze_recipe.py /workspace/scratch/2d09435bab69/work/e5_v2/scripts/t1gr_e5_freeze_recipe.py
--- /workspace/scratch/2d09435bab69/tmp/review/e5_failure/scripts/t1gr_e5_freeze_recipe.py	2026-08-20 02:55:26.000000000 +0800
+++ /workspace/scratch/2d09435bab69/work/e5_v2/scripts/t1gr_e5_freeze_recipe.py	2026-08-20 13:13:41.965226884 +0800
@@ -18,7 +18,7 @@
     utc_now, validate_e2_evidence, validate_e4_evidence, validate_training_spec, parse_utc,
 )
 
-SCRIPT_VERSION="t1gr-e5-freeze-recipe-hardened-v1"
+SCRIPT_VERSION="t1gr-e5-v2-freeze-recipe-hardened-v2"
 
 def git_head(repo:Path)->str:
     try:
@@ -33,7 +33,7 @@
     repo=ROOT.resolve(strict=True)
     secp=ensure_repo_input(repo,"config/t1gr_e5_security_policy.json","config")
     if sha256_file(secp)!=FROZEN_E5_SECURITY_POLICY_SHA256: fail("E5_SECURITY_POLICY_SHA_DRIFT")
-    sec=read_json_bounded(secp,1<<20,"t1gr-e5-security-policy-v1")
+    sec=read_json_bounded(secp,1<<20,"t1gr-e5-security-policy-v2")
     tsp=ensure_repo_input(repo,"config/t1gr_e5_training_spec.frozen.json","config")
     if sha256_file(tsp)!=FROZEN_E5_TRAINING_SPEC_SHA256: fail("E5_FROZEN_TRAINING_SPEC_SHA_DRIFT")
     forensic_p=ensure_repo_input(repo,"reports/step4_t1gr/zip_forensic_public.json","reports/step4_t1gr")
@@ -41,7 +41,7 @@
     e4f_p=ensure_repo_input(repo,"reports/step4_t1gr/e4_split_freeze_public.json","reports/step4_t1gr")
     e4v_p=ensure_repo_input(repo,"reports/step4_t1gr/e4_seal_verification_public.json","reports/step4_t1gr")
     td_p=ensure_private_input(Path(a.train_dev_access),repo)
-    out=ensure_public_output(repo,"reports/step4_t1gr/e5_step1_recipe_public.json",sec["public_output_prefix"])
+    out=ensure_public_output(repo,"reports/step4_t1gr/e5_v2_step1_recipe_public.json",sec["public_output_prefix"])
     zp=Path(a.formal_zip).expanduser().resolve(strict=False)
     ck=Path(a.base_checkpoint).expanduser().resolve(strict=False)
     if not zp.is_file(): fail("FORMAL_ZIP_NOT_FOUND")
@@ -56,7 +56,7 @@
         e4f=read_json_bounded(e4f_p,int(sec["max_public_json_bytes"]))
         e4v=read_json_bounded(e4v_p,int(sec["max_public_json_bytes"]))
         td=read_json_bounded(td_p,int(sec["max_private_json_bytes"]))
-        spec=read_json_bounded(tsp,int(sec["max_public_json_bytes"]),"t1gr-e5-training-spec-v1")
+        spec=read_json_bounded(tsp,int(sec["max_public_json_bytes"]),"t1gr-e5-training-spec-v2")
         validate_e2_evidence(forensic,taxonomy)
         validate_training_spec(spec)
         td_sha=sha256_file(td_p,deadline)
@@ -118,6 +118,8 @@
             "train_args":spec["train_args"],
             "eval_args":spec["eval_args"],
             "runtime":spec["runtime"],
+            "optimizer_adjudication":spec["review_freeze"]["optimizer_adjudication"],
+            "evaluation_detection_cap":spec["review_freeze"]["evaluation_detection_cap"],
             "view_policy":{
                 "mode":"COPY_ONLY",
                 "modalities":"VISIBLE_PLUS_LABELS",
diff -ruN /workspace/scratch/2d09435bab69/tmp/review/e5_failure/scripts/t1gr_e5_preflight.py /workspace/scratch/2d09435bab69/work/e5_v2/scripts/t1gr_e5_preflight.py
--- /workspace/scratch/2d09435bab69/tmp/review/e5_failure/scripts/t1gr_e5_preflight.py	2026-08-20 02:55:26.000000000 +0800
+++ /workspace/scratch/2d09435bab69/work/e5_v2/scripts/t1gr_e5_preflight.py	2026-08-20 13:13:42.149226882 +0800
@@ -11,48 +11,22 @@
 )
 from multimodal.t1gr_e5_core import (
  FROZEN_E5_SECURITY_POLICY_SHA256,FROZEN_E5_TRAINING_SPEC_SHA256,
- SCHEMA_PREFLIGHT,SCHEMA_RECIPE,SCHEMA_VIEW_PUBLIC,compare_environment,environment_probe,payload_ok,verify_view_tree
+ SCHEMA_PREFLIGHT,SCHEMA_RECIPE,SCHEMA_VIEW_PUBLIC,build_seeded_model,
+ compare_environment,environment_probe,payload_ok,verify_view_tree
 )
 
-SCRIPT_VERSION="t1gr-e5-step1-preflight-hardened-v1"
-
-def build_model(checkpoint:Path,recipe:dict):
-    try:
-        import torch
-        from ultralytics.nn.tasks import DetectionModel,yaml_model_load
-    except Exception:fail("E5_MODEL_IMPORT_FAIL")
-    try:
-        d=yaml_model_load(recipe["model_yaml"]);d["nc"]=12;d["end2end"]=True
-        model=DetectionModel(d,ch=3,nc=12,verbose=False)
-        ckpt=torch.load(checkpoint,map_location="cpu",weights_only=False)
-        src=ckpt["model"].float().state_dict()
-        dst=model.state_dict()
-        compatible={k:v for k,v in src.items() if k in dst and tuple(v.shape)==tuple(dst[k].shape)}
-        if not compatible:fail("E5_PRETRAIN_TRANSFER_EMPTY")
-        model.load_state_dict(compatible,strict=False)
-    except KeyError:fail("E5_CHECKPOINT_MODEL_KEY_MISSING")
-    except Exception as e:
-        if hasattr(e,"code"):raise
-        fail("E5_MODEL_BUILD_OR_LOAD_FAIL")
-    head=model.model[-1]
-    return model,{
-        "physical_nc":int(getattr(head,"nc",-1)),
-        "end2end":bool(getattr(head,"end2end",getattr(model,"end2end",False))),
-        "destination_state_keys":len(dst),"source_state_keys":len(src),
-        "transferred_state_keys":len(compatible),
-        "transfer_fraction_of_destination":len(compatible)/max(1,len(dst)),
-    }
+SCRIPT_VERSION="t1gr-e5-v2-step1-preflight-hardened-v2"
 
 def run(a):
     repo=ROOT.resolve(strict=True)
     secp=ensure_repo_input(repo,"config/t1gr_e5_security_policy.json","config")
     if sha256_file(secp)!=FROZEN_E5_SECURITY_POLICY_SHA256: fail("E5_SECURITY_POLICY_SHA_DRIFT")
-    sec=read_json_bounded(secp,1<<20,"t1gr-e5-security-policy-v1")
-    rp=ensure_repo_input(repo,"reports/step4_t1gr/e5_step1_recipe_public.json","reports/step4_t1gr")
-    vpubp=ensure_repo_input(repo,"reports/step4_t1gr/e5_step1_view_public.json","reports/step4_t1gr")
+    sec=read_json_bounded(secp,1<<20,"t1gr-e5-security-policy-v2")
+    rp=ensure_repo_input(repo,"reports/step4_t1gr/e5_v2_step1_recipe_public.json","reports/step4_t1gr")
+    vpubp=ensure_repo_input(repo,"reports/step4_t1gr/e5_v2_step1_view_public.json","reports/step4_t1gr")
     td_p=ensure_private_input(Path(a.train_dev_access),repo)
     vm_p=ensure_private_input(Path(a.view_manifest),repo)
-    out=ensure_public_output(repo,"reports/step4_t1gr/e5_step1_preflight_public.json",sec["public_output_prefix"])
+    out=ensure_public_output(repo,"reports/step4_t1gr/e5_v2_step1_preflight_public.json",sec["public_output_prefix"])
     ck=Path(a.base_checkpoint).expanduser().resolve(strict=False)
     if not ck.is_file():fail("BASE_CHECKPOINT_NOT_FOUND")
     deadline=Deadline(float(a.timeout_seconds or sec["view_verify_timeout_seconds"]))
@@ -67,7 +41,7 @@
         ck_sha=sha256_file(ck,deadline)
         if ck_sha!=recipe["base_checkpoint_sha256"]:fail("E5_CHECKPOINT_SHA_DRIFT")
         env=environment_probe();compare_environment(env,recipe["environment"])
-        _,model_info=build_model(ck,recipe)
+        _,model_info=build_seeded_model(ck,recipe)
         if model_info["physical_nc"]!=12:fail("E5_PHYSICAL_HEAD_NC_FAIL")
         if model_info["end2end"] is not True:fail("E5_HEAD_END2END_FAIL")
         request_fp=sha256_json({"script":SCRIPT_VERSION,"recipe":sha256_file(rp,deadline),
diff -ruN /workspace/scratch/2d09435bab69/tmp/review/e5_failure/scripts/t1gr_e5_run_step1.py /workspace/scratch/2d09435bab69/work/e5_v2/scripts/t1gr_e5_run_step1.py
--- /workspace/scratch/2d09435bab69/tmp/review/e5_failure/scripts/t1gr_e5_run_step1.py	2026-08-20 02:55:26.000000000 +0800
+++ /workspace/scratch/2d09435bab69/work/e5_v2/scripts/t1gr_e5_run_step1.py	2026-08-20 13:13:53.433226755 +0800
@@ -2,7 +2,7 @@
 """Hardened Step1 RGB smoke/formal trainer. Scientific args come ONLY from frozen recipe."""
 from __future__ import annotations
 
-import argparse,csv,json,os,shutil,sys,traceback
+import argparse,json,os,sys
 from pathlib import Path
 
 ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
@@ -13,12 +13,12 @@
 from multimodal.t1gr_e5_core import (
  FROZEN_E5_SECURITY_POLICY_SHA256,FROZEN_E5_TRAINING_SPEC_SHA256,
  SCHEMA_PREFLIGHT,SCHEMA_RECIPE,SCHEMA_RUN,SCHEMA_VIEW_PUBLIC,
- compare_environment,effective_args_mismatch,environment_probe,optimizer_fingerprint,
+ build_seeded_model,compare_environment,effective_args_mismatch,environment_probe,optimizer_fingerprint,
  parse_utc,payload_ok,private_umask,results_csv_epoch_count,ultralytics_offline_guard,
- utc_now,verify_view_tree,wall_clock_watchdog
+ state_dict_sha256,utc_now,verify_view_tree,wall_clock_watchdog,write_private_failure_report
 )
 
-SCRIPT_VERSION="t1gr-e5-step1-trainer-hardened-v1"
+SCRIPT_VERSION="t1gr-e5-v2-step1-trainer-hardened-v2"
 
 def private_run_root(path:Path,repo:Path)->Path:
     p=path.expanduser().resolve(strict=False)
@@ -31,24 +31,6 @@
         except OSError: fail("E5_RUN_ROOT_PERMISSION_HARDEN_FAIL")
     return p
 
-def build_model(checkpoint:Path,recipe:dict):
-    try:
-        import torch
-        from ultralytics.nn.tasks import DetectionModel,yaml_model_load
-    except Exception:fail("E5_MODEL_IMPORT_FAIL")
-    try:
-        d=yaml_model_load(recipe["model_yaml"]);d["nc"]=12;d["end2end"]=True
-        model=DetectionModel(d,ch=3,nc=12,verbose=False)
-        ckpt=torch.load(checkpoint,map_location="cpu",weights_only=False)
-        src=ckpt["model"].float().state_dict();dst=model.state_dict()
-        compatible={k:v for k,v in src.items() if k in dst and tuple(v.shape)==tuple(dst[k].shape)}
-        if not compatible:fail("E5_PRETRAIN_TRANSFER_EMPTY")
-        model.load_state_dict(compatible,strict=False)
-        return model,len(compatible),len(dst)
-    except Exception as e:
-        if hasattr(e,"code"):raise
-        fail("E5_MODEL_BUILD_OR_LOAD_FAIL")
-
 def expected_effective(recipe:dict,mode:str)->dict:
     d=dict(recipe["train_args"])
     if mode=="smoke":
@@ -56,36 +38,39 @@
     d.update(recipe["eval_args"])
     return d
 
-def validate_smoke(smoke:dict,recipe_sha:str,view_sha:str,ck_sha:str):
+def validate_smoke(smoke:dict,recipe_sha:str,view_sha:str,ck_sha:str,initial_state_sha:str):
     if smoke.get("schema")!=SCHEMA_RUN or smoke.get("mode")!="smoke" or smoke.get("run_gate_passed") is not True:
         fail("E5_SMOKE_REPORT_NOT_PASS")
     if smoke.get("recipe_public_sha256")!=recipe_sha or smoke.get("view_manifest_private_sha256")!=view_sha:
         fail("E5_SMOKE_PROVENANCE_DRIFT")
     if smoke.get("base_checkpoint_sha256")!=ck_sha:fail("E5_SMOKE_CHECKPOINT_DRIFT")
+    if smoke.get("model_initial_state_sha256")!=initial_state_sha:fail("E5_SMOKE_INITIAL_STATE_DRIFT")
     if smoke.get("formal_step1_authorized_after_smoke") is not True:fail("E5_SMOKE_DID_NOT_AUTHORIZE_FORMAL")
 
 def run(a):
     repo=ROOT.resolve(strict=True)
     secp=ensure_repo_input(repo,"config/t1gr_e5_security_policy.json","config")
     if sha256_file(secp)!=FROZEN_E5_SECURITY_POLICY_SHA256: fail("E5_SECURITY_POLICY_SHA_DRIFT")
-    sec=read_json_bounded(secp,1<<20,"t1gr-e5-security-policy-v1")
-    rp=ensure_repo_input(repo,"reports/step4_t1gr/e5_step1_recipe_public.json","reports/step4_t1gr")
-    vpubp=ensure_repo_input(repo,"reports/step4_t1gr/e5_step1_view_public.json","reports/step4_t1gr")
-    pfp=ensure_repo_input(repo,"reports/step4_t1gr/e5_step1_preflight_public.json","reports/step4_t1gr")
+    sec=read_json_bounded(secp,1<<20,"t1gr-e5-security-policy-v2")
+    rp=ensure_repo_input(repo,"reports/step4_t1gr/e5_v2_step1_recipe_public.json","reports/step4_t1gr")
+    vpubp=ensure_repo_input(repo,"reports/step4_t1gr/e5_v2_step1_view_public.json","reports/step4_t1gr")
+    pfp=ensure_repo_input(repo,"reports/step4_t1gr/e5_v2_step1_preflight_public.json","reports/step4_t1gr")
     td_p=ensure_private_input(Path(a.train_dev_access),repo)
     vm_p=ensure_private_input(Path(a.view_manifest),repo)
     recipe=read_json_bounded(rp,int(sec["max_public_json_bytes"]),SCHEMA_RECIPE)
     vpub=read_json_bounded(vpubp,int(sec["max_public_json_bytes"]),SCHEMA_VIEW_PUBLIC)
     pre=read_json_bounded(pfp,int(sec["max_public_json_bytes"]),SCHEMA_PREFLIGHT)
     td=read_json_bounded(td_p,int(sec["max_private_json_bytes"]))
+    if int(sec.get("private_failure_traceback_max_bytes",-1))!=int(recipe["runtime"]["private_traceback_max_bytes"]):
+        fail("E5_PRIVATE_TRACEBACK_POLICY_DRIFT")
     if not all(payload_ok(x) for x in (recipe,vpub,pre,td)):fail("E5_RUN_INPUT_INTEGRITY_FAIL")
     if pre.get("preflight_gate_passed") is not True or pre.get("smoke_authorized") is not True:fail("E5_PREFLIGHT_NOT_PASS")
     if a.mode not in {"smoke","formal"}:fail("E5_RUN_MODE_INVALID")
     run_root=private_run_root(Path(a.run_root),repo)
-    run_name="STEP1_RGB_SMOKE" if a.mode=="smoke" else "STEP1_RGB_BASELINE"
+    run_name="STEP1_RGB_SMOKE_V2" if a.mode=="smoke" else "STEP1_RGB_BASELINE_V2"
     run_dir=run_root/run_name
-    out_rel=("reports/step4_t1gr/e5_step1_smoke_public.json" if a.mode=="smoke"
-             else "reports/step4_t1gr/e5_step1_formal_run_public.json")
+    out_rel=("reports/step4_t1gr/e5_v2_step1_smoke_public.json" if a.mode=="smoke"
+             else "reports/step4_t1gr/e5_v2_step1_formal_run_public.json")
     out=ensure_public_output(repo,out_rel,sec["public_output_prefix"])
     ck=Path(a.base_checkpoint).expanduser().resolve(strict=False)
     if not ck.is_file():fail("BASE_CHECKPOINT_NOT_FOUND")
@@ -102,19 +87,29 @@
         if pre.get("recipe_public_sha256")!=recipe_sha or pre.get("view_manifest_private_sha256")!=view_sha:
             fail("E5_RUN_PREFLIGHT_PROVENANCE_DRIFT")
         if pre.get("base_checkpoint_sha256")!=ck_sha:fail("E5_RUN_PREFLIGHT_CHECKPOINT_DRIFT")
+        pre_model=pre.get("model_preflight") or {}
+        initial_state_sha=pre_model.get("model_initial_state_sha256")
+        untransferred_state_sha=pre_model.get("untransferred_initial_state_sha256")
+        if not isinstance(initial_state_sha,str) or len(initial_state_sha)!=64:
+            fail("E5_PREFLIGHT_INITIAL_STATE_PIN_MISSING")
+        if not isinstance(untransferred_state_sha,str) or len(untransferred_state_sha)!=64:
+            fail("E5_PREFLIGHT_UNTRANSFERRED_STATE_PIN_MISSING")
 
         smoke_sha=None
         if a.mode=="formal":
-            sp=ensure_repo_input(repo,"reports/step4_t1gr/e5_step1_smoke_public.json","reports/step4_t1gr")
+            sp=ensure_repo_input(repo,"reports/step4_t1gr/e5_v2_step1_smoke_public.json","reports/step4_t1gr")
             smoke=read_json_bounded(sp,int(sec["max_public_json_bytes"]),SCHEMA_RUN)
             if not payload_ok(smoke):fail("E5_SMOKE_REPORT_INTEGRITY_FAIL")
-            validate_smoke(smoke,recipe_sha,view_sha,ck_sha)
+            validate_smoke(smoke,recipe_sha,view_sha,ck_sha,initial_state_sha)
+            if smoke.get("untransferred_initial_state_sha256")!=untransferred_state_sha:
+                fail("E5_SMOKE_UNTRANSFERRED_STATE_DRIFT")
             smoke_sha=sha256_file(sp,deadline)
 
         run_root_binding=sha256_json(str(run_root).casefold() if os.name=="nt" else str(run_root))
         request_fp=sha256_json({"script":SCRIPT_VERSION,"mode":a.mode,"recipe":recipe_sha,
                                "view":view_sha,"preflight":sha256_file(pfp,deadline),
                                "checkpoint":ck_sha,"smoke_report":smoke_sha,
+                               "model_initial_state_sha256":initial_state_sha,
                                "run_root_binding":run_root_binding})
         # A completed same-request public report is idempotent; do not retrain.
         from multimodal.t1gr_secure_io import check_existing_output
@@ -122,6 +117,7 @@
         if existing is not None:
             obj,sh=existing
             if obj.get("run_gate_passed") is not True:fail("E5_EXISTING_RUN_NOT_PASS")
+            if obj.get("model_initial_state_sha256")!=initial_state_sha:fail("E5_EXISTING_INITIAL_STATE_DRIFT")
             if not run_dir.is_dir(): fail("E5_EXISTING_RUN_DIRECTORY_MISSING")
             last0=run_dir/"weights"/"last.pt";args0=run_dir/"args.yaml";csv0=run_dir/"results.csv"
             if not last0.is_file() or not args0.is_file() or not csv0.is_file(): fail("E5_EXISTING_RUN_ARTIFACT_MISSING")
@@ -137,9 +133,6 @@
         if os.name!="nt":
             try: os.chmod(run_root,0o700)
             except OSError: fail("E5_RUN_ROOT_PERMISSION_HARDEN_FAIL")
-        if os.name!="nt":
-            try: os.chmod(run_root,0o700)
-            except OSError: fail("E5_RUN_ROOT_PERMISSION_HARDEN_FAIL")
         start=utc_now()
         if not parse_utc(recipe["e4_freeze_timestamp_utc"]) < parse_utc(start):
             fail("E5_E4_FREEZE_NOT_BEFORE_TRAINING")
@@ -152,7 +145,11 @@
         except Exception:fail("E5_TRAINER_IMPORT_FAIL")
         if str(ultralytics.__version__)!=recipe["environment"]["ultralytics_version"]:fail("E5_TRAIN_ULTRALYTICS_VERSION_DRIFT")
 
-        model,transfer_count,dst_count=build_model(ck,recipe)
+        model,model_info=build_seeded_model(ck,recipe)
+        if model_info["model_initial_state_sha256"]!=initial_state_sha:
+            fail("E5_MODEL_INITIAL_STATE_NOT_REPRODUCIBLE")
+        if model_info["untransferred_initial_state_sha256"]!=untransferred_state_sha:
+            fail("E5_MODEL_UNTRANSFERRED_STATE_NOT_REPRODUCIBLE")
         head=model.model[-1]
         if int(getattr(head,"nc",-1))!=12:fail("E5_TRAIN_PHYSICAL_HEAD_NC_FAIL")
         if bool(getattr(head,"end2end",getattr(model,"end2end",False))) is not True:fail("E5_TRAIN_HEAD_MODE_FAIL")
@@ -170,12 +167,15 @@
         expected=expected_effective(recipe,a.mode)
         expected.update({"resume":False,"profile":False,"verbose":True,"pretrained":False,"exist_ok":False})
         optimizer_capture={}
+        training_start_state={}
         offline_state={}
         permission_state={}
+        phase="trainer_setup"
         try:
             with ultralytics_offline_guard(bypass_amp_download_check=bool(expected["amp"])) as og, private_umask() as pg:
                 offline_state.update(og); permission_state.update(pg)
                 trainer=DetectionTrainer(overrides=overrides)
+                phase="effective_args_preflight"
                 mm=effective_args_mismatch(trainer.args,expected)
                 if mm:fail("E5_EFFECTIVE_ARGS_PREFLIGHT_MISMATCH",f"count={len(mm)}")
                 trainer.model=model;trainer.model.args=trainer.args
@@ -187,6 +187,10 @@
                     if actual_workers!=int(expected["workers"]): fail("E5_EFFECTIVE_WORKERS_DRIFT")
                 def optimizer_cb(t):
                     runtime_check(t)
+                    if not training_start_state:
+                        training_start_state["sha256"]=state_dict_sha256(t.model.state_dict())
+                        if training_start_state["sha256"]!=initial_state_sha:
+                            fail("E5_TRAINING_START_STATE_DRIFT")
                     if not optimizer_capture:
                         optimizer_capture.update(optimizer_fingerprint(t.optimizer))
                         expected_name=str(recipe["train_args"]["optimizer"]).lower()
@@ -195,14 +199,26 @@
                 trainer.add_callback("on_train_batch_end",runtime_check)
                 trainer.add_callback("on_train_epoch_end",runtime_check)
                 trainer.add_callback("on_train_start",optimizer_cb)
+                phase="trainer_train"
                 with wall_clock_watchdog(timeout,"E5_TRAINING_TIMEOUT"):
                     trainer.train()
+                phase="post_train_runtime_check"
                 runtime_check(trainer)
-        except Exception:
-            # Keep private partial run for diagnosis; never publish PASS.
+                if not training_start_state:fail("E5_TRAINING_START_STATE_NOT_CAPTURED")
+        except BaseException as exc:
+            # Keep a private traceback and partial run for diagnosis; public stderr stays sanitized.
+            try:
+                write_private_failure_report(
+                    run_dir,exc,phase,int(recipe["runtime"]["private_traceback_max_bytes"])
+                )
+            except Exception:
+                pass
             try:
                 if run_dir.exists():
-                    (run_dir/"E5_INCOMPLETE.txt").write_text("Formal PASS not issued. Inspect locally; remove directory explicitly before rerun.\n",encoding="utf-8")
+                    (run_dir/"E5_INCOMPLETE.txt").write_text(
+                        "E5 v2 PASS not issued. Inspect E5_PRIVATE_FAILURE.json locally; "
+                        "archive or remove this run directory explicitly before rerun.\n",encoding="utf-8"
+                    )
             except Exception:pass
             raise
 
@@ -229,8 +245,13 @@
             "train_count":vr["train_count"],"dev_count":vr["dev_count"],
             "ids_commitments":recipe["ids_commitments"],
             "physical_head_nc":12,"head_end2end":True,
-            "pretrained_transfer_key_count":transfer_count,
-            "pretrained_destination_key_count":dst_count,
+            "pretrained_transfer_key_count":model_info["transferred_state_keys"],
+            "pretrained_destination_key_count":model_info["destination_state_keys"],
+            "untransferred_state_key_count":model_info["untransferred_state_keys"],
+            "model_initialization_effective_seed":model_info["model_initialization_effective_seed"],
+            "model_initial_state_sha256":model_info["model_initial_state_sha256"],
+            "untransferred_initial_state_sha256":model_info["untransferred_initial_state_sha256"],
+            "training_start_state_sha256":training_start_state.get("sha256"),
             "optimizer":optimizer_capture,
             "external_network_integrations":offline_state,
             "private_artifact_permissions":permission_state,
diff -ruN /workspace/scratch/2d09435bab69/tmp/review/e5_failure/scripts/t1gr_e5_v2_regression_gate.py /workspace/scratch/2d09435bab69/work/e5_v2/scripts/t1gr_e5_v2_regression_gate.py
--- /workspace/scratch/2d09435bab69/tmp/review/e5_failure/scripts/t1gr_e5_v2_regression_gate.py	1970-01-01 08:00:00.000000000 +0800
+++ /workspace/scratch/2d09435bab69/work/e5_v2/scripts/t1gr_e5_v2_regression_gate.py	2026-08-20 13:08:31.401230381 +0800
@@ -0,0 +1,50 @@
+#!/usr/bin/env python3
+"""Dependency-light regression gate for the E5 v2 bundle."""
+from __future__ import annotations
+
+import importlib.util
+import inspect
+import json
+import sys
+import traceback
+from pathlib import Path
+
+ROOT = Path(__file__).resolve().parents[1]
+TEST_FILE = ROOT / "tests" / "test_t1gr_e5_hardened.py"
+
+
+def main() -> int:
+    spec = importlib.util.spec_from_file_location("t1gr_e5_v2_tests", TEST_FILE)
+    if spec is None or spec.loader is None:
+        print(json.dumps({"status": "FAIL", "error": "TEST_MODULE_LOAD_FAIL"}))
+        return 2
+    module = importlib.util.module_from_spec(spec)
+    spec.loader.exec_module(module)
+    tests = [
+        (name, fn) for name, fn in vars(module).items()
+        if name.startswith("test_") and callable(fn) and len(inspect.signature(fn).parameters) == 0
+    ]
+    failures = []
+    for name, fn in sorted(tests):
+        try:
+            fn()
+        except BaseException as exc:
+            failures.append({
+                "test": name,
+                "exception_type": type(exc).__name__,
+                "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
+            })
+    result = {
+        "schema": "t1gr-e5-v2-regression-gate-v1",
+        "status": "PASS" if not failures else "FAIL",
+        "passed": len(tests) - len(failures),
+        "total": len(tests),
+        "failures": failures,
+    }
+    print(json.dumps(result, ensure_ascii=False, indent=2))
+    return 0 if not failures else 2
+
+
+if __name__ == "__main__":
+    raise SystemExit(main())
+

Binary files /workspace/scratch/2d09435bab69/tmp/review/e5_failure/src/multimodal/__pycache__/t1gr_e5_core.cpython-312.pyc and /workspace/scratch/2d09435bab69/work/e5_v2/src/multimodal/__pycache__/t1gr_e5_core.cpython-312.pyc differ
Binary files /workspace/scratch/2d09435bab69/tmp/review/e5_failure/src/multimodal/__pycache__/t1gr_secure_io.cpython-312.pyc and /workspace/scratch/2d09435bab69/work/e5_v2/src/multimodal/__pycache__/t1gr_secure_io.cpython-312.pyc differ
diff -ruN /workspace/scratch/2d09435bab69/tmp/review/e5_failure/src/multimodal/t1gr_e5_core.py /workspace/scratch/2d09435bab69/work/e5_v2/src/multimodal/t1gr_e5_core.py
--- /workspace/scratch/2d09435bab69/tmp/review/e5_failure/src/multimodal/t1gr_e5_core.py	2026-08-20 02:55:26.000000000 +0800
+++ /workspace/scratch/2d09435bab69/work/e5_v2/src/multimodal/t1gr_e5_core.py	2026-08-20 13:14:04.409226632 +0800
@@ -12,6 +12,7 @@
 import contextlib
 import threading
 import _thread
+import traceback
 from pathlib import Path
 from typing import Any
 
@@ -26,13 +27,13 @@
 ]
 CLASS_NAME_MAP = {str(i): name for i, name in enumerate(CLASS_NAMES)}
 
-SCHEMA_RECIPE = "t1gr-e5-step1-recipe-public-v1"
-SCHEMA_VIEW_PRIVATE = "t1gr-e5-step1-view-private-v1"
-SCHEMA_VIEW_PUBLIC = "t1gr-e5-step1-view-public-v1"
-SCHEMA_PREFLIGHT = "t1gr-e5-step1-preflight-public-v1"
-SCHEMA_RUN = "t1gr-e5-step1-run-public-v1"
-SCHEMA_EVAL = "t1gr-e5-step1-eval-public-v1"
-SCHEMA_FINAL = "t1gr-e5-final-audit-public-v1"
+SCHEMA_RECIPE = "t1gr-e5-v2-step1-recipe-public-v2"
+SCHEMA_VIEW_PRIVATE = "t1gr-e5-v2-step1-view-private-v2"
+SCHEMA_VIEW_PUBLIC = "t1gr-e5-v2-step1-view-public-v2"
+SCHEMA_PREFLIGHT = "t1gr-e5-v2-step1-preflight-public-v2"
+SCHEMA_RUN = "t1gr-e5-v2-step1-run-public-v2"
+SCHEMA_EVAL = "t1gr-e5-v2-step1-eval-public-v2"
+SCHEMA_FINAL = "t1gr-e5-v2-final-audit-public-v2"
 
 E4_FREEZE_SCHEMA = "t1gr-e4-split-freeze-public-v1"
 E4_VERIFY_SCHEMA = "t1gr-e4-seal-verification-public-v1"
@@ -41,8 +42,8 @@
 FORENSIC_SCHEMA = "t1gr-zip-forensic-public-v1"
 TAXONOMY_SCHEMA = "t1gr-label-error-taxonomy-public-v1"
 
-FROZEN_E5_TRAINING_SPEC_SHA256 = "01be2a9443d068fca13ce7b4fdaee481fb16de9ca4bc12ad2ef756d64cdfd32e"
-FROZEN_E5_SECURITY_POLICY_SHA256 = "656fcacb191aa7e85d463c1abb46203cd3a2eb2347cfec72dd09bb7ae4a18c52"
+FROZEN_E5_TRAINING_SPEC_SHA256 = "a6b83b6c28a2b794978d787bbc4e75278b41fd6556a56c026e15341830d335f2"
+FROZEN_E5_SECURITY_POLICY_SHA256 = "4b105a66e1a06b3cb5adb0c1446054487ae379060e70b84eb08936bb3d1d1b98"
 
 REQUIRED_TRAIN_ARGS = (
     "epochs", "batch", "imgsz", "patience", "optimizer", "lr0", "lrf", "momentum",
@@ -58,6 +59,7 @@
 REQUIRED_RUNTIME = (
     "device", "smoke_epochs", "smoke_timeout_seconds", "formal_timeout_seconds",
     "eval_timeout_seconds", "lock_wait_seconds", "lock_stale_seconds",
+    "private_traceback_max_bytes",
 )
 
 HEX64 = re.compile(r"^[0-9a-f]{64}$")
@@ -207,9 +209,9 @@
 
 
 def validate_training_spec(spec: dict) -> None:
-    if spec.get("schema") != "t1gr-e5-training-spec-v1":
+    if spec.get("schema") != "t1gr-e5-training-spec-v2":
         fail("E5_TRAINING_SPEC_SCHEMA_FAIL")
-    if spec.get("status") != "REVIEWED_FROZEN":
+    if spec.get("status") != "REVIEWED_FROZEN_V2":
         fail("E5_TRAINING_SPEC_NOT_REVIEWED")
     train = require_dict(spec.get("train_args"), "E5_TRAIN_ARGS_MISSING")
     eva = require_dict(spec.get("eval_args"), "E5_EVAL_ARGS_MISSING")
@@ -218,11 +220,8 @@
     require_keys(eva, REQUIRED_EVAL_ARGS, "E5_EVAL_ARGS_UNRESOLVED")
     require_keys(runtime, REQUIRED_RUNTIME, "E5_RUNTIME_UNRESOLVED")
 
-    allowed_opt={"sgd","musgd","adam","adamax","adamw","nadam","radam","rmsprop","auto"}
-    if not isinstance(train["optimizer"],str) or train["optimizer"].lower() not in allowed_opt:
-        fail("E5_OPTIMIZER_INVALID")
-    if str(train["optimizer"]).lower() == "auto":
-        fail("E5_OPTIMIZER_AUTO_FORBIDDEN")
+    if str(train.get("optimizer")).lower() != "musgd":
+        fail("E5_V2_OPTIMIZER_MUST_BE_ADJUDICATED_MUSGD")
     if train["deterministic"] is not True:
         fail("E5_DETERMINISTIC_REQUIRED")
     if train["end2end"] is not True:
@@ -231,6 +230,8 @@
         fail("E5_CORE_TRAIN_POLICY_FAIL")
     if eva["split"] != "val":
         fail("E5_EVAL_MUST_BE_DEV_VAL")
+    if int(eva.get("max_det", -1)) != 100:
+        fail("E5_EVAL_MAX_DET_MUST_BE_100")
     if spec.get("architecture") != "yolo26s" or spec.get("model_yaml") != "yolo26s.yaml":
         fail("E5_MODEL_ARCH_DRIFT")
     if int(spec.get("num_classes", -1)) != 12:
@@ -253,8 +254,9 @@
         _need_numeric(eva, k, minv=0.0, maxv=1.0)
     _need_numeric(eva, "max_det", minv=1, integer=True)
     for k in ("smoke_epochs","smoke_timeout_seconds","formal_timeout_seconds","eval_timeout_seconds",
-              "lock_wait_seconds","lock_stale_seconds"):
+              "lock_wait_seconds","lock_stale_seconds","private_traceback_max_bytes"):
         _need_numeric(runtime, k, minv=1.0)
+    _need_numeric(runtime, "private_traceback_max_bytes", minv=65536, integer=True)
     if int(runtime["smoke_epochs"]) != 1:
         fail("E5_SMOKE_EPOCHS_MUST_BE_ONE")
     if not isinstance(runtime["device"], (str, int)) or str(runtime["device"]).strip() == "":
@@ -268,6 +270,38 @@
     if any(not isinstance(eva[k],bool) for k in bool_eval): fail("E5_EVAL_SPEC_BOOL_TYPE_FAIL")
     if train["copy_paste_mode"] not in {"flip","mixup"}: fail("E5_COPY_PASTE_MODE_INVALID")
 
+    review = require_dict(spec.get("review_freeze"), "E5_V2_REVIEW_FREEZE_MISSING")
+    adjudication = require_dict(review.get("optimizer_adjudication"), "E5_V2_OPTIMIZER_ADJUDICATION_MISSING")
+    require_keys(adjudication, (
+        "decision", "selected_optimizer", "framework_auto_would_select",
+        "train_sample_count", "nominal_batch", "nbs", "epochs",
+        "framework_iteration_formula", "framework_estimated_iterations",
+        "auto_threshold_iterations", "not_auto_equivalent", "rationale",
+    ), "E5_V2_OPTIMIZER_ADJUDICATION_UNRESOLVED")
+    expected_iterations = math.ceil(
+        int(adjudication["train_sample_count"]) /
+        max(int(adjudication["nominal_batch"]), int(adjudication["nbs"]))
+    ) * int(adjudication["epochs"])
+    if (
+        adjudication["decision"] != "KEEP_PROJECT_FROZEN_MUSGD"
+        or adjudication["selected_optimizer"] != train["optimizer"]
+        or adjudication["framework_auto_would_select"] != "AdamW"
+        or int(adjudication["train_sample_count"]) != 1504
+        or int(adjudication["nominal_batch"]) != int(train["batch"])
+        or int(adjudication["nbs"]) != int(train["nbs"])
+        or int(adjudication["epochs"]) != int(train["epochs"])
+        or adjudication["framework_iteration_formula"] != "ceil(train_sample_count/max(nominal_batch,nbs))*epochs"
+        or int(adjudication["framework_estimated_iterations"]) != expected_iterations
+        or expected_iterations != 1920
+        or int(adjudication["auto_threshold_iterations"]) != 10000
+        or adjudication["not_auto_equivalent"] is not True
+        or not isinstance(adjudication["rationale"], str)
+        or not adjudication["rationale"].strip()
+    ):
+        fail("E5_V2_OPTIMIZER_ADJUDICATION_DRIFT")
+    if int(review.get("evaluation_detection_cap", -1)) != int(eva["max_det"]):
+        fail("E5_V2_EVALUATION_CAP_DRIFT")
+
 
 def zip_modality(name: str) -> str | None:
     parts = [x for x in name.replace("\\", "/").split("/") if x]
@@ -342,6 +376,138 @@
     }
 
 
+def state_dict_sha256(state: dict[str, Any], keys: list[str] | None = None) -> str:
+    """Hash tensor names, metadata and bytes in a stable key order."""
+    try:
+        import torch
+    except Exception:
+        fail("E5_MODEL_HASH_IMPORT_FAIL")
+    selected = sorted(state) if keys is None else sorted(keys)
+    h = hashlib.sha256()
+    for name in selected:
+        if name not in state:
+            fail("E5_MODEL_HASH_KEY_MISSING")
+        tensor = state[name]
+        if not isinstance(tensor, torch.Tensor):
+            fail("E5_MODEL_HASH_NON_TENSOR_STATE")
+        value = tensor.detach().cpu().contiguous()
+        metadata = json.dumps(
+            {"name": name, "dtype": str(value.dtype), "shape": list(value.shape)},
+            sort_keys=True, separators=(",", ":"), ensure_ascii=False,
+        ).encode("utf-8")
+        byte_view_source = value.reshape(1) if value.ndim == 0 else value
+        raw = byte_view_source.view(torch.uint8).reshape(-1).numpy().tobytes()
+        h.update(len(metadata).to_bytes(8, "big")); h.update(metadata)
+        h.update(len(raw).to_bytes(8, "big")); h.update(raw)
+    return h.hexdigest()
+
+
+def build_seeded_model(checkpoint: Path, recipe: dict) -> tuple[Any, dict]:
+    """Seed before nc=12 head construction, transfer the pinned checkpoint and hash the exact initial state."""
+    try:
+        import torch
+        from ultralytics.nn.tasks import DetectionModel, yaml_model_load
+        from ultralytics.utils import RANK
+        from ultralytics.utils.torch_utils import init_seeds
+    except Exception:
+        fail("E5_MODEL_IMPORT_FAIL")
+    if int(RANK) != -1:
+        fail("E5_DISTRIBUTED_RANK_FORBIDDEN")
+    effective_seed = int(recipe["train_args"]["seed"]) + 1 + int(RANK)
+    try:
+        init_seeds(effective_seed, deterministic=bool(recipe["train_args"]["deterministic"]))
+        definition = yaml_model_load(recipe["model_yaml"])
+        definition["nc"] = 12
+        definition["end2end"] = True
+        model = DetectionModel(definition, ch=3, nc=12, verbose=False)
+        checkpoint_obj = torch.load(checkpoint, map_location="cpu", weights_only=False)
+        source = checkpoint_obj["model"].float().state_dict()
+        destination = model.state_dict()
+        compatible = {
+            key: value for key, value in source.items()
+            if key in destination and tuple(value.shape) == tuple(destination[key].shape)
+        }
+        if not compatible:
+            fail("E5_PRETRAIN_TRANSFER_EMPTY")
+        model.load_state_dict(compatible, strict=False)
+        initialized = model.state_dict()
+        untransferred = sorted(set(initialized) - set(compatible))
+        head = model.model[-1]
+        return model, {
+            "physical_nc": int(getattr(head, "nc", -1)),
+            "end2end": bool(getattr(head, "end2end", getattr(model, "end2end", False))),
+            "destination_state_keys": len(destination),
+            "source_state_keys": len(source),
+            "transferred_state_keys": len(compatible),
+            "transfer_fraction_of_destination": len(compatible) / max(1, len(destination)),
+            "untransferred_state_keys": len(untransferred),
+            "model_initialization_rank": int(RANK),
+            "model_initialization_effective_seed": effective_seed,
+            "model_initial_state_sha256": state_dict_sha256(initialized),
+            "untransferred_initial_state_sha256": state_dict_sha256(initialized, untransferred),
+        }
+    except KeyError:
+        fail("E5_CHECKPOINT_MODEL_KEY_MISSING")
+    except Exception as exc:
+        if hasattr(exc, "code"):
+            raise
+        fail("E5_MODEL_BUILD_OR_LOAD_FAIL")
+
+
+def write_private_failure_report(run_dir: Path, exc: BaseException, phase: str, max_bytes: int) -> Path:
+    """Persist the full local traceback in the private run directory; public stderr remains sanitized."""
+    if not isinstance(max_bytes, int) or max_bytes < 65536:
+        fail("E5_PRIVATE_TRACEBACK_LIMIT_INVALID")
+    run_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
+    path = run_dir / "E5_PRIVATE_FAILURE.json"
+    temp = run_dir / f".E5_PRIVATE_FAILURE.{os.getpid()}.tmp"
+    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
+    csv_path = run_dir / "results.csv"
+    completed_rows = None
+    if csv_path.is_file():
+        try:
+            with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
+                completed_rows = sum(1 for _ in csv.DictReader(stream))
+        except OSError:
+            completed_rows = None
+    report = {
+        "schema": "t1gr-e5-v2-private-failure-v1",
+        "recorded_at_utc": utc_now(),
+        "phase": str(phase),
+        "exception_type": type(exc).__name__,
+        "exception_message": str(exc)[: max_bytes // 8],
+        "traceback": tb,
+        "partial_artifacts": {
+            "results_csv_present": csv_path.is_file(),
+            "completed_result_rows": completed_rows,
+            "args_yaml_present": (run_dir / "args.yaml").is_file(),
+            "last_pt_present": (run_dir / "weights" / "last.pt").is_file(),
+            "best_pt_present": (run_dir / "weights" / "best.pt").is_file(),
+        },
+        "public_pass_issued": False,
+    }
+    data = json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8")
+    if len(data) > max_bytes:
+        report["traceback"] = tb.encode("utf-8")[: max_bytes // 2].decode("utf-8", errors="replace")
+        report["traceback_truncated"] = True
+        data = json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8")
+    if len(data) > max_bytes:
+        fail("E5_PRIVATE_TRACEBACK_REPORT_TOO_LARGE")
+    try:
+        with temp.open("xb") as stream:
+            stream.write(data); stream.flush(); os.fsync(stream.fileno())
+        if os.name != "nt":
+            os.chmod(temp, 0o600)
+        os.replace(temp, path)
+    except OSError:
+        try:
+            temp.unlink(missing_ok=True)
+        except OSError:
+            pass
+        fail("E5_PRIVATE_TRACEBACK_WRITE_FAIL")
+    return path
+
+
 def environment_probe() -> dict:
     try:
         import platform

Binary files /workspace/scratch/2d09435bab69/tmp/review/e5_failure/tests/__pycache__/test_t1gr_e5_hardened.cpython-312.pyc and /workspace/scratch/2d09435bab69/work/e5_v2/tests/__pycache__/test_t1gr_e5_hardened.cpython-312.pyc differ
diff -ruN /workspace/scratch/2d09435bab69/tmp/review/e5_failure/tests/test_t1gr_e5_hardened.py /workspace/scratch/2d09435bab69/work/e5_v2/tests/test_t1gr_e5_hardened.py
--- /workspace/scratch/2d09435bab69/tmp/review/e5_failure/tests/test_t1gr_e5_hardened.py	2026-08-20 02:55:26.000000000 +0800
+++ /workspace/scratch/2d09435bab69/work/e5_v2/tests/test_t1gr_e5_hardened.py	2026-08-20 13:14:10.937226558 +0800
@@ -7,7 +7,7 @@
 
 def candidate():
     x=json.loads((ROOT/"config/t1gr_e5_training_spec.candidate.json").read_text())
-    x["status"]="REVIEWED_FROZEN"
+    x["status"]="REVIEWED_FROZEN_V2"
     return x
 
 def with_payload(o,fp="x"):
@@ -53,7 +53,7 @@
 def test_optimizer_auto_rejected():
     x=candidate();x["train_args"]["optimizer"]="auto"
     try:validate_training_spec(x)
-    except GateError as e:assert e.code=="E5_OPTIMIZER_AUTO_FORBIDDEN"
+    except GateError as e:assert e.code=="E5_V2_OPTIMIZER_MUST_BE_ADJUDICATED_MUSGD"
     else:raise AssertionError
 
 def test_null_workers_rejected():
@@ -124,7 +124,7 @@
     assert candidate()["eval_args"]["conf"]==0.001
 
 def test_eval_max_det_explicit():
-    assert candidate()["eval_args"]["max_det"]==300
+    assert candidate()["eval_args"]["max_det"]==100
 
 def test_effective_args_mismatch():
     class X: a=1
@@ -146,7 +146,9 @@
 
 def test_security_policy_no_holdout_input():
     p=json.loads((ROOT/"config/t1gr_e5_security_policy.json").read_text())
+    assert p["schema"]=="t1gr-e5-security-policy-v2"
     assert p["final_holdout_sealed_artifact_is_not_an_E5_input"] is True
+    assert p["private_failure_traceback_max_bytes"]==candidate()["runtime"]["private_traceback_max_bytes"]
 
 def test_runner_has_no_scientific_cli_overrides():
     s=(ROOT/"scripts/t1gr_e5_run_step1.py").read_text()
@@ -155,7 +157,7 @@
 
 def test_runner_formal_requires_fixed_smoke():
     s=(ROOT/"scripts/t1gr_e5_run_step1.py").read_text()
-    assert 'reports/step4_t1gr/e5_step1_smoke_public.json' in s
+    assert 'reports/step4_t1gr/e5_v2_step1_smoke_public.json' in s
     assert 'if a.mode=="formal"' in s
 
 def test_eval_no_holdout_arg():
@@ -242,6 +244,12 @@
     assert FROZEN_E5_TRAINING_SPEC_SHA256 in s
     assert FROZEN_E5_SECURITY_POLICY_SHA256 in s
 
+def test_frozen_config_sha_pins_match_files():
+    spec_hash=hashlib.sha256((ROOT/"config/t1gr_e5_training_spec.frozen.json").read_bytes()).hexdigest()
+    security_hash=hashlib.sha256((ROOT/"config/t1gr_e5_security_policy.json").read_bytes()).hexdigest()
+    assert spec_hash==FROZEN_E5_TRAINING_SPEC_SHA256
+    assert security_hash==FROZEN_E5_SECURITY_POLICY_SHA256
+
 def test_operational_public_inputs_fixed():
     names={
       "t1gr_e5_freeze_recipe.py","t1gr_e5_build_rgb_view.py","t1gr_e5_preflight.py",
@@ -275,3 +283,67 @@
 def test_environment_pins_ultralytics_source_hashes():
     s=(ROOT/"src/multimodal/t1gr_e5_core.py").read_text()
     assert "ultralytics_source_sha256" in s and "trainer_py" in s and "default_yaml" in s
+
+def test_v2_optimizer_adjudication_is_explicit_and_truthful():
+    x=candidate();a=x["review_freeze"]["optimizer_adjudication"]
+    assert a["decision"]=="KEEP_PROJECT_FROZEN_MUSGD"
+    assert a["selected_optimizer"]=="MuSGD"
+    assert a["framework_auto_would_select"]=="AdamW"
+    assert a["framework_estimated_iterations"]==1920
+    assert a["not_auto_equivalent"] is True
+
+def test_v2_optimizer_adjudication_drift_rejected():
+    x=candidate();x["review_freeze"]["optimizer_adjudication"]["framework_estimated_iterations"]=30080
+    try:validate_training_spec(x)
+    except GateError as e:assert e.code=="E5_V2_OPTIMIZER_ADJUDICATION_DRIFT"
+    else:raise AssertionError
+
+def test_v2_max_det_drift_rejected():
+    x=candidate();x["eval_args"]["max_det"]=300
+    try:validate_training_spec(x)
+    except GateError as e:assert e.code=="E5_EVAL_MAX_DET_MUST_BE_100"
+    else:raise AssertionError
+
+def test_model_seed_occurs_before_detection_model_construction():
+    s=(ROOT/"src/multimodal/t1gr_e5_core.py").read_text()
+    start=s.index("def build_seeded_model")
+    body=s[start:s.index("def write_private_failure_report",start)]
+    assert body.index("init_seeds(effective_seed") < body.index("DetectionModel(definition")
+
+def test_preflight_and_runner_share_seeded_model_builder():
+    for name in ("t1gr_e5_preflight.py","t1gr_e5_run_step1.py"):
+        s=(ROOT/"scripts"/name).read_text()
+        assert "build_seeded_model(ck,recipe)" in s
+
+def test_initial_state_sha_pinned_across_gates():
+    runner=(ROOT/"scripts/t1gr_e5_run_step1.py").read_text()
+    audit=(ROOT/"scripts/t1gr_e5_final_audit.py").read_text()
+    assert "E5_MODEL_INITIAL_STATE_NOT_REPRODUCIBLE" in runner
+    assert "E5_TRAINING_START_STATE_DRIFT" in runner
+    assert "initial_state_preflight_smoke_pin" in audit
+    assert "initial_state_preflight_formal_pin" in audit
+    assert "formal_training_start_state_pin" in audit
+
+def test_private_failure_report_is_local_and_bounded():
+    with tempfile.TemporaryDirectory() as d:
+        try:raise RuntimeError("synthetic-private-trace")
+        except RuntimeError as exc:
+            p=write_private_failure_report(Path(d),exc,"synthetic_phase",65536)
+        x=json.loads(p.read_text(encoding="utf-8"))
+        assert x["phase"]=="synthetic_phase"
+        assert x["exception_type"]=="RuntimeError"
+        assert "synthetic-private-trace" in x["traceback"]
+        assert x["public_pass_issued"] is False
+
+def test_runner_catches_base_exception_for_private_traceback():
+    s=(ROOT/"scripts/t1gr_e5_run_step1.py").read_text()
+    assert "except BaseException as exc" in s
+    assert "E5_PRIVATE_FAILURE.json" in (ROOT/"src/multimodal/t1gr_e5_core.py").read_text()
+
+def test_v2_output_names_do_not_collide_with_v1():
+    operational=(ROOT/"scripts/t1gr_e5_run_step1.py").read_text()
+    assert "STEP1_RGB_SMOKE_V2" in operational and "STEP1_RGB_BASELINE_V2" in operational
+    for name in ("t1gr_e5_freeze_recipe.py","t1gr_e5_build_rgb_view.py","t1gr_e5_preflight.py",
+                 "t1gr_e5_run_step1.py","t1gr_e5_eval_step1.py","t1gr_e5_final_audit.py"):
+        s=(ROOT/"scripts"/name).read_text()
+        assert "e5_v2" in s

```

