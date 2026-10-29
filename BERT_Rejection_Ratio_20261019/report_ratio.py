"""Report the fixed 0/10/25/50-percent detector-fit ratio grid.

Run only after TEST_COMPLETE.json. No model inference, detector fitting or
threshold selection is performed. All outputs are derived reporting artifacts.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import html
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

PROJECT = Path(__file__).resolve().parent
PREVIOUS = PROJECT.parent / "BERT_Mixed_Rejection_20261019"
REJECTION = PROJECT.parent / "BERT_Rejection_Study_20261019"
sys.path.insert(0, str(REJECTION))
sys.path.insert(0, str(PREVIOUS))
from report_rejection import SEEDS, POLICIES, POLICY_NAMES, booleans, evaluate_predictions, htable, image_tag
from report_typo_rejection import CONDITIONS, CONDITION_NAMES, aggregate
from report_mixed_rejection import manifest_hashes, sha256

ARMS = ("clean_reference", "ratio_00", "ratio_10", "ratio_25", "ratio_50")
GRID = ARMS[1:]
CANDIDATES = ("ratio_10", "ratio_25")
REFERENCES = ("ratio_00", "ratio_50")
HISTORY = {"clean_reference": "clean_only", "ratio_00": "clean_duplicate", "ratio_50": "clean_plus_typo"}
RATIOS = {"clean_reference": 0., "ratio_00": 0., "ratio_10": .10, "ratio_25": .25, "ratio_50": .50}
ARM_NAMES = {"clean_reference": "历史clean（15k）", "ratio_00": "0%（30k）", "ratio_10": "10%（30k）", "ratio_25": "25%（30k）", "ratio_50": "50%（30k）"}
FOCUS_POLICY = "is_acceptance_95"
METRICS = ("in_scope_false_rejection_rate", "in_scope_correct_and_accepted_rate", "in_scope_wrong_and_accepted_rate",
           "oos_false_acceptance_rate", "oos_f1", "oos_recall", "in_scope_acceptance", "combined_151_accuracy")
NAMES = {"in_scope_false_rejection_rate": "正常误拒", "in_scope_correct_and_accepted_rate": "正常正确且接受",
         "in_scope_wrong_and_accepted_rate": "正常错分类放行", "oos_false_acceptance_rate": "OOS错误接受",
         "oos_f1": "OOS F1", "oos_recall": "OOS recall", "in_scope_acceptance": "IS接受率", "combined_151_accuracy": "系统Accuracy"}
STATES = ("accepted_correct", "accepted_wrong", "rejected_correct", "rejected_wrong")


def load_results(output):
    manifests = [PROJECT/"INPUTS_FROZEN.json", output/"DEVELOPMENT_FROZEN.json", output/"TEST_COMPLETE.json"]
    hashes = {}
    for path in manifests:
        hashes.update(manifest_hashes(json.loads(path.read_text(encoding="utf-8"))))
    metric_path = output/"metrics_by_seed.csv"
    metrics = pd.read_csv(metric_path, float_precision="round_trip")
    keys = ["arm", "seed", "condition", "score", "policy"]
    if len(metrics) != 90 or metrics.duplicated(keys).any():
        raise ValueError("Expected 90 unique ratio test settings")
    for name, choices in (("arm",ARMS),("seed",SEEDS),("condition",CONDITIONS),("score",("mahalanobis",)),("policy",POLICIES)):
        if set(metrics[name]) != set(choices):
            raise ValueError(f"Unexpected {name}")
    history_path = PREVIOUS/"outputs/metrics_by_seed.csv"
    historical = pd.read_csv(history_path, float_precision="round_trip").set_index(["arm","seed","condition","policy"])
    inputs = [metric_path, history_path, *manifests]
    identity = ["row_id","original_text","true_label","label_id","is_oos"]
    shared_cols = identity + ["text","changed","modified_tokens","predicted_label_id"]
    frames, canonical, shared = {}, None, {}
    for arm in ARMS:
        frames[arm] = {}
        for seed in SEEDS:
            frames[arm][seed] = {}
            for condition in CONDITIONS:
                path = output/arm/f"seed{seed}"/f"{condition}_predictions.csv"
                frame = pd.read_csv(path, float_precision="round_trip")
                needed = set(shared_cols + ["mahalanobis_score"]+[f"mahalanobis_{p}_reject" for p in POLICIES])
                if not needed.issubset(frame) or frame[list(needed)].isna().any().any() or frame.row_id.duplicated().any():
                    raise ValueError(f"Invalid predictions: {path}")
                frame["is_oos"], frame["changed"] = booleans(frame.is_oos), booleans(frame.changed)
                if canonical is None:
                    canonical = frame[identity].copy()
                else:
                    if set(frame.row_id) != set(canonical.row_id):
                        raise ValueError("Original-text rows differ")
                    frame = frame.set_index("row_id").loc[canonical.row_id].reset_index()
                    if not frame[identity].equals(canonical):
                        raise ValueError("Original identities differ")
                if arm == ARMS[0]:
                    shared[(seed,condition)] = frame[shared_cols].copy()
                elif not frame[shared_cols].equals(shared[(seed,condition)]):
                    raise ValueError("Classifier or perturbed texts differ across ratio arms")
                labels, oos, prediction = frame.label_id.to_numpy(), frame.is_oos.to_numpy(), frame.predicted_label_id.to_numpy()
                unique, counts = np.unique(labels[~oos], return_counts=True)
                if (len(frame)!=5500 or oos.sum()!=1000 or not np.all(labels[oos]==-1)
                        or not np.array_equal(unique,np.arange(150)) or not np.all(counts==30)):
                    raise ValueError("Expected 4500 IS (150x30) and 1000 OOS")
                if not np.array_equal(frame.changed, frame.text!=frame.original_text):
                    raise ValueError("Changed flags differ from texts")
                values = frame.mahalanobis_score.to_numpy(dtype=float)
                if not np.isfinite(values).all():
                    raise ValueError("Invalid scores")
                closed = float((prediction[~oos]==labels[~oos]).mean())
                macro = float(f1_score(labels[~oos],prediction[~oos],labels=np.arange(150),average="macro"))
                for policy in POLICIES:
                    mask = metrics.arm.eq(arm)&metrics.seed.eq(seed)&metrics.condition.eq(condition)&metrics.policy.eq(policy)
                    row = metrics.loc[mask].iloc[0]
                    frozen = metrics[metrics.arm.eq(arm)&metrics.seed.eq(seed)&metrics.policy.eq(policy)]
                    if frozen.threshold.nunique(dropna=False)!=1 or pd.isna(row.threshold):
                        raise ValueError("Threshold changed across test conditions")
                    rejected = booleans(frame[f"mahalanobis_{policy}_reject"])
                    frame[f"mahalanobis_{policy}_reject"] = rejected
                    if not np.array_equal(rejected,values>=float(row.threshold)):
                        raise ValueError("Threshold and decisions disagree")
                    calculated = evaluate_predictions(labels,prediction,oos,values,rejected)
                    calculated.update(closed_set_accuracy=closed,closed_set_macro_f1=macro)
                    for metric,value in calculated.items():
                        if not np.isclose(value,float(row[metric]),rtol=1e-8,atol=1e-10,equal_nan=True):
                            raise ValueError(f"Metric mismatch: {arm}/{seed}/{condition}/{policy}/{metric}")
                    metrics.loc[mask,"in_scope_wrong_and_accepted_rate"] = float((~oos & ~rejected & (prediction!=labels)).sum()/4500)
                    metrics.loc[mask,"oos_false_acceptance_rate"] = float((oos & ~rejected).sum()/1000)
                    if arm in HISTORY:
                        old = historical.loc[(HISTORY[arm],seed,condition,policy)]
                        if float(row.threshold)!=float(old.threshold):
                            raise ValueError("Saved historical threshold changed")
                        for metric in calculated:
                            if not np.isclose(float(row[metric]),float(old[metric]),rtol=1e-8,atol=1e-10,equal_nan=True):
                                raise ValueError("Saved reference metrics changed")
                if arm in HISTORY:
                    old_path = PREVIOUS/"outputs"/HISTORY[arm]/f"seed{seed}"/f"{condition}_predictions.csv"
                    old = pd.read_csv(old_path,float_precision="round_trip").set_index("row_id").loc[frame.row_id].reset_index()
                    if not np.array_equal(old.predicted_label_id,prediction) or not np.array_equal(old.mahalanobis_score,values):
                        raise ValueError("Copied reference predictions or scores differ")
                    inputs.append(old_path)
                frames[arm][seed][condition] = frame
                inputs.append(path)
    for _, group in metrics.groupby(["seed","condition"]):
        if group.closed_set_accuracy.nunique()!=1 or group.closed_set_macro_f1.nunique()!=1:
            raise ValueError("Closed accuracy/F1 must be exactly invariant")
    for path in inputs:
        expected = hashes.get(str(path.resolve()))
        if expected and sha256(path)!=expected:
            raise ValueError(f"Consumed frozen input changed: {path}")
    return metrics,frames,inputs


def states_and_transitions(frames):
    states, transitions = [], []
    for arm in ARMS:
        for seed in SEEDS:
            for condition in CONDITIONS:
                frame=frames[arm][seed][condition]
                scope=~frame.is_oos.to_numpy()
                correct=frame.predicted_label_id.to_numpy()==frame.label_id.to_numpy()
                for policy in POLICIES:
                    rejected=frame[f"mahalanobis_{policy}_reject"].to_numpy()
                    row=dict(arm=arm,seed=seed,condition=condition,score="mahalanobis",policy=policy,n_is=4500)
                    for name,mask in zip(STATES,(~rejected&correct,~rejected&~correct,rejected&correct,rejected&~correct)):
                        row[name+"_n"]=int((mask&scope).sum())
                        row[name+"_rate_over_all_is"]=row[name+"_n"]/4500
                    states.append(row)
    for candidate in CANDIDATES:
        for reference in REFERENCES:
            for seed in SEEDS:
                for condition in CONDITIONS:
                    target,source=frames[candidate][seed][condition],frames[reference][seed][condition]
                    scope=~target.is_oos.to_numpy()
                    correct=target.predicted_label_id.to_numpy()==target.label_id.to_numpy()
                    for policy in POLICIES:
                        before=source[f"mahalanobis_{policy}_reject"].to_numpy()
                        after=target[f"mahalanobis_{policy}_reject"].to_numpy()
                        for stratum,mask in (("all_is",scope),("head_correct",scope&correct),("head_wrong",scope&~correct)):
                            n=int(mask.sum())
                            row=dict(candidate=candidate,reference=reference,comparison=candidate+"_minus_"+reference,
                                     seed=seed,condition=condition,score="mahalanobis",policy=policy,stratum=stratum,n_is=4500,stratum_n=n,
                                     source_accepted_n=int((mask&~before).sum()),source_rejected_n=int((mask&before).sum()))
                            for name,event in (("accept_accept",~before&~after),("accept_reject",~before&after),
                                               ("reject_accept",before&~after),("reject_reject",before&after)):
                                row[name+"_n"]=int((mask&event).sum())
                                row[name+"_rate_over_all_is"]=row[name+"_n"]/4500
                                row[name+"_rate_over_stratum"]=row[name+"_n"]/n if n else np.nan
                            row["net_rejection_change_n"]=row["accept_reject_n"]-row["reject_accept_n"]
                            row["net_rejection_change_rate_over_all_is"]=row["net_rejection_change_n"]/4500
                            transitions.append(row)
    return pd.DataFrame(states),pd.DataFrame(transitions)


def derived_tables(summary,metrics):
    table=summary.set_index(["arm","condition","score","policy"])
    # Equal 4500-IS denominators permit exact integer count differences before
    # division, avoiding a spurious positive denominator from round-off at zero.
    counts=metrics.groupby(["arm","condition","score","policy"])["in_scope_false_rejections"].sum()
    guardrails,retention=[] ,[]
    for candidate in CANDIDATES:
        for policy in POLICIES:
            for reference in ("ratio_00","ratio_50","clean_reference"):
                local=[]
                for metric in ("oos_f1","oos_recall","in_scope_acceptance"):
                    before,after=(float(table.loc[(arm,"clean","mahalanobis",policy)][metric+"_mean"]) for arm in (reference,candidate))
                    local.append(dict(candidate=candidate,reference=reference,condition="clean",score="mahalanobis",policy=policy,
                                      metric=metric,reference_mean=before,candidate_mean=after,delta=after-before,delta_pp=100*(after-before),
                                      mean_non_decline=after>=before,interpretation="improvement_relative_to_50" if reference=="ratio_50" else "strict_preservation"))
                passed=all(row["mean_non_decline"] for row in local)
                for row in local:row["all_three_pass"]=passed
                guardrails.extend(local)
            for condition in CONDITIONS:
                totals={arm:int(counts.loc[(arm,condition,"mahalanobis",policy)]) for arm in ARMS}
                means={arm:totals[arm]/(len(SEEDS)*4500) for arm in ARMS}
                denominator_count=totals["ratio_00"]-totals["ratio_50"]
                numerator_count=totals["ratio_00"]-totals[candidate]
                denominator=denominator_count/(len(SEEDS)*4500)
                retention.append(dict(candidate=candidate,condition=condition,score="mahalanobis",policy=policy,
                                      baseline="ratio_00",benefit_reference="ratio_50",baseline_fr_mean=means["ratio_00"],
                                      candidate_fr_mean=means[candidate],ratio50_fr_mean=means["ratio_50"],
                                      historical_clean_reference_fr_mean=means["clean_reference"],
                                      numerator=numerator_count/(len(SEEDS)*4500),denominator=denominator,
                                      defined=denominator_count>0,benefit_retention_percent=100*numerator_count/denominator_count if denominator_count>0 else np.nan,
                                      formula="100*(mean_FR_ratio00-mean_FR_candidate)/(mean_FR_ratio00-mean_FR_ratio50); denominator must be positive"))
    return pd.DataFrame(guardrails),pd.DataFrame(retention)


def counts_to_metrics(counts):
    tp,fp,accepted_correct,accepted_wrong=(counts[...,i,:] for i in range(4))
    return np.stack((fp/4500,accepted_correct/4500,accepted_wrong/4500,1-tp/1000,
                     2*tp/(tp+fp+1000),tp/1000,1-fp/4500,(tp+accepted_correct)/5500),axis=-2)


def paired_bootstrap(frames,repeats=2000,random_seed=20261021,batch_size=64):
    if repeats<2 or batch_size<1:raise ValueError("Invalid bootstrap settings")
    first=frames[ARMS[0]][SEEDS[0]]["clean"]
    labels,oos=first.label_id.to_numpy(),first.is_oos.to_numpy()
    strata=[np.flatnonzero(labels==label) for label in np.unique(labels)]
    events=np.zeros((5,3,3,2,4,5500),dtype=float)
    for ai,arm in enumerate(ARMS):
        for si,seed in enumerate(SEEDS):
            for ci,condition in enumerate(CONDITIONS):
                frame=frames[arm][seed][condition]
                correct=(frame.predicted_label_id.to_numpy()==labels)&~oos
                for pi,policy in enumerate(POLICIES):
                    rejected=frame[f"mahalanobis_{policy}_reject"].to_numpy()
                    events[ai,si,ci,pi]=np.stack((rejected&oos,rejected&~oos,~rejected&correct,~rejected&~correct&~oos))
    point=counts_to_metrics(events.sum(axis=-1)[...,None]).mean(axis=1)[...,0]
    draws=np.empty((*point.shape,repeats))
    rng=np.random.default_rng(random_seed)
    for start in range(0,repeats,batch_size):
        size=min(batch_size,repeats-start)
        sample=np.concatenate([rng.choice(indices,size=(size,len(indices)),replace=True) for indices in strata],axis=1)
        weights=np.bincount((sample+5500*np.arange(size)[:,None]).ravel(),minlength=size*5500).reshape(size,5500).astype(float)
        counts=(events.reshape(-1,5500)@weights.T).reshape(*events.shape[:-1],size)
        draws[...,start:start+size]=counts_to_metrics(counts).mean(axis=1)
    rows=[]
    for candidate in CANDIDATES:
        for reference in REFERENCES:
            a,b=ARMS.index(candidate),ARMS.index(reference)
            for ci,condition in enumerate(CONDITIONS):
                for pi,policy in enumerate(POLICIES):
                    for mi,metric in enumerate(METRICS):
                        low,high=np.quantile(draws[a,ci,pi,mi]-draws[b,ci,pi,mi],[.025,.975])
                        delta=point[a,ci,pi,mi]-point[b,ci,pi,mi]
                        rows.append(dict(candidate=candidate,reference=reference,comparison=candidate+"_minus_"+reference,
                                         condition=condition,score="mahalanobis",policy=policy,metric=metric,estimate_delta=delta,
                                         ci_low=low,ci_high=high,estimate_delta_pp=100*delta,ci_low_pp=100*low,ci_high_pp=100*high,
                                         bootstrap_repeats=repeats,bootstrap_seed=random_seed,n_is=4500,n_oos=1000,n_fixed_models=3,
                                         scope="fixed models, cached features, nested subset, detector fits and clean-validation thresholds",
                                         resampling="150 IS intent strata x30 and 1000 OOS; shared original-text weights across arms, seeds, conditions and policies"))
    return pd.DataFrame(rows)


def create_plots(summary,retention,intervals,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    table=summary.set_index(["arm","condition","score","policy"])
    x=np.array([0,10,25,50])
    colors=("#397c91","#c27b35","#6e669f")
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    panels=(("in_scope_false_rejection_rate","IS false rejection (%)"),("in_scope_correct_and_accepted_rate","IS correct + accepted (%)"),
            ("in_scope_wrong_and_accepted_rate","IS wrong-head accepted / all IS (%)"),("oos_false_acceptance_rate","OOS false acceptance (%)"))
    for ax,(metric,title) in zip(axes.flat,panels):
        for color,condition in zip(colors,CONDITIONS):
            rows=[table.loc[(arm,condition,"mahalanobis",FOCUS_POLICY)] for arm in GRID]
            ax.errorbar(x,[100*r[metric+"_mean"] for r in rows],yerr=[100*r[metric+"_sd"] for r in rows],
                        marker="o",capsize=3,color=color,label=condition.replace("typo_","").title())
        ax.set_xticks(x);ax.set_xlabel("Planned noisy views in the fixed 30k fit pool (%)")
        ax.set_ylabel(title);ax.spines[["right","top"]].set_visible(False)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc="outside lower center",ncol=3)
    fig.suptitle("Full predeclared ratio grid | fixed classifier | clean-validation IS >=95% | mean +/- sample SD")
    curve=output/"ratio_tradeoff_curves.png";fig.savefig(curve,dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4.8),constrained_layout=True)
    for ax,metric in zip(axes,("oos_f1","oos_recall","in_scope_acceptance")):
        rows=[table.loc[(arm,"clean","mahalanobis",FOCUS_POLICY)] for arm in GRID]
        ax.errorbar(x,[100*r[metric+"_mean"] for r in rows],yerr=[100*r[metric+"_sd"] for r in rows],marker="o",capsize=3,color=colors[0],label="Fixed 30k ratio grid")
        history=table.loc[("clean_reference","clean","mahalanobis",FOCUS_POLICY)]
        ax.axhline(100*history[metric+"_mean"],linestyle="--",color=colors[1],label="Historical clean 15k mean")
        ax.set_xticks(x);ax.set_xlabel("Planned noisy ratio (%)");ax.set_ylabel("Clean test (%)");ax.set_title(metric.replace("_"," "))
        ax.spines[["right","top"]].set_visible(False)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="outside lower center",ncol=2)
    fig.suptitle("Clean guardrails: 50% improvement and 0% / historical preservation are different questions")
    clean=output/"ratio_clean_guardrails.png";fig.savefig(clean,dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,8),constrained_layout=True)
    for ax,candidate in zip(axes,CANDIDATES):
        rows=intervals[(intervals.candidate==candidate)&(intervals.policy==FOCUS_POLICY)
                       & (((intervals.condition=="clean")&intervals.metric.isin(("oos_f1","oos_recall","in_scope_acceptance")))
                          | ((intervals.condition!="clean")&intervals.metric.isin(("in_scope_false_rejection_rate","in_scope_wrong_and_accepted_rate","oos_false_acceptance_rate"))))]
        # Show both references on identical endpoint rows, keeping the full CSV available.
        labels=[]
        for ref,color,offset in zip(REFERENCES,colors,(-.14,.14)):
            part=rows[rows.reference==ref]
            y=np.arange(len(part))+offset
            ax.hlines(y,part.ci_low_pp,part.ci_high_pp,color=color,lw=2)
            ax.scatter(part.estimate_delta_pp,y,color=color,s=25,label="minus "+("0%" if ref=="ratio_00" else "50%"))
            labels=[r.condition.replace("typo_","").title()+": "+{"oos_f1":"OOS F1","oos_recall":"OOS recall","in_scope_acceptance":"IS acceptance","in_scope_false_rejection_rate":"IS false rejection","in_scope_wrong_and_accepted_rate":"IS wrong accepted","oos_false_acceptance_rate":"OOS false accept"}[r.metric] for r in part.itertuples()]
        ax.axvline(0,color="#80909a",ls="--",lw=1);ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis()
        ax.set_title(candidate[-2:]+"% candidate: both reference comparisons");ax.set_xlabel("Candidate minus reference (pp)")
        ax.spines[["right","top"]].set_visible(False)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="outside lower center",ncol=2)
    fig.suptitle("Paired 95% conditional intervals; error rates: negative is better; clean guardrails: positive is better")
    paired=output/"ratio_paired_intervals.png";fig.savefig(paired,dpi=180);plt.close(fig)
    return curve,clean,paired


def fmt(row,name,multiplier=100):
    return f"{row[name+'_mean']*multiplier:.2f} ± {row[name+'_sd']*multiplier:.2f}"


def evidence_summary(evidence,inputs):
    parts=[]
    audit=evidence.get("data_audit.json")
    if audit:
        if not all(audit["checks"].values()):raise ValueError("Independent allocation audit failed")
        parts.append(f"<p>独立数据审计复现{audit['checked_selection_rows']:,}条来源排名与{audit['checked_fit_manifest_rows']:,}条拟合清单，"
                     "核对逐类light/medium嵌套选择、原标签、每源权重和实际改变覆盖。没有新增相对验证/test的精确或规范化碰撞；"
                     "原有重叠继续披露，不能等同于不存在污染。</p>")
    verification=evidence.get("verification.json")
    if verification:
        if verification["status"]!="PASS" or verification["metrics_sha256"]!=sha256(inputs[0]):
            raise ValueError("Verification failed or refers to different frozen metrics")
        solves=verification["direct_covariance_solve_checks"]
        parts.append(f"<p>独立实验核验PASS：{verification['verified_test_settings']}个test设置、{verification['verified_validation_settings']}个validation运行点、"
                     f"{verification['validation_candidates_checked']:,}个候选阈值；{verification['new_detectors_independently_recomputed']}个新detector的中心、协方差与Ledoit–Wolf统计独立重算，"
                     "三组旧参考逐seed的模型、阈值、分数和预测均核对一致。"
                     f"{sum(len(r['row_ids']) for r in solves)}个CLS逐一对150中心直接解协方差，最大距离误差{max(r['max_abs_distance_error'] for r in solves):.2e}。"
                     f"派生表独立核验状态：{html.escape(str(verification['report_checks']['status']))}。</p>")
    sensitivity=evidence.get("duplicate_sensitivity.json")
    if sensitivity:
        parts.append("<p>对全部比例、seed、条件与两策略统一排除原始row_id "
                     + html.escape(str(sensitivity["excluded_original_row_ids"]))
                     + f" 后，方法差值最大绝对变化为{sensitivity['max_absolute_change_in_method_difference_pp']:.8f} pp。"
                     "该敏感性仅重算固定输出，没有重拟合或重选阈值，不能撤销历史训练/验证重叠和test曝光。</p>")
    return "".join(parts)


def build_html(summary,guardrails,retention,intervals,plots,data,inputs,evidence,repeats,seed,validation):
    table=summary.set_index(["arm","condition","score","policy"])
    lead=[]
    benefit_rows=[]
    for candidate in CANDIDATES:
        bits=[]
        for condition in CONDITIONS[1:]:
            row=table.loc[(candidate,condition,"mahalanobis",FOCUS_POLICY)]
            rr=retention[(retention.candidate==candidate)&(retention.condition==condition)&(retention.policy==FOCUS_POLICY)].iloc[0]
            benefit=f"{rr.benefit_retention_percent:.2f}%" if rr.defined else "未定义（分母不为正）"
            bits.append(f"{CONDITION_NAMES[condition]}误拒{100*row.in_scope_false_rejection_rate_mean:.2f}%，保留50%档相对0%档误拒收益的{benefit}")
            benefit_rows.append([ARM_NAMES[candidate],CONDITION_NAMES[condition],f"{100*rr.baseline_fr_mean:.3f}%",f"{100*rr.candidate_fr_mean:.3f}%",f"{100*rr.ratio50_fr_mean:.3f}%",benefit])
        lead.append(ARM_NAMES[candidate]+"："+"；".join(bits)+"。")
    main_rows,clean_rows,full_rows,data_rows=[],[],[],[]
    for arm in ARMS:
        info=data["arms"][arm]
        data_rows.append([ARM_NAMES[arm],info["fit_rows"],f"{100*info['planned_noisy_ratio']:.0f}%",info["selected_noisy_views"],
                          f"{info['selected_light']} / {info['selected_medium']}",info["actual_changed_rows"],f"{100*info['actual_changed_pool_fraction']:.2f}%"])
        for condition in CONDITIONS:
            row=table.loc[(arm,condition,"mahalanobis",FOCUS_POLICY)]
            main_rows.append([ARM_NAMES[arm],CONDITION_NAMES[condition],*[fmt(row,m) for m in METRICS[:4]],fmt(row,"oos_f1")])
        row=table.loc[(arm,"clean","mahalanobis",FOCUS_POLICY)]
        clean_rows.append([ARM_NAMES[arm],*[fmt(row,m) for m in ("oos_f1","oos_recall","in_scope_acceptance")]])
    guardrail_rows=[]
    verdicts=[]
    selected=guardrails[guardrails.policy==FOCUS_POLICY]
    for candidate in CANDIDATES:
        for reference in ("ratio_50","ratio_00","clean_reference"):
            group=selected[(selected.candidate==candidate)&(selected.reference==reference)]
            passed=bool(group.all_three_pass.iloc[0])
            verdicts.append(f"{ARM_NAMES[candidate]}相对{ARM_NAMES[reference]}：三项均值"+("均未下降" if passed else "存在下降"))
            for row in group.itertuples():
                ci=intervals[(intervals.candidate==candidate)&(intervals.reference==reference)&(intervals.condition=="clean")
                             &(intervals.policy==FOCUS_POLICY)&(intervals.metric==row.metric)]
                ci_text="只作均值描述" if ci.empty else f"[{ci.iloc[0].ci_low_pp:+.6f}, {ci.iloc[0].ci_high_pp:+.6f}]"
                guardrail_rows.append([ARM_NAMES[candidate],ARM_NAMES[reference],NAMES[row.metric],f"{row.delta_pp:+.4f}",ci_text,
                                       "未下降" if row.mean_non_decline else "下降", "三项均未下降" if passed else "未全部保留"])
    clean_ci=intervals[(intervals.condition=="clean")&(intervals.policy==FOCUS_POLICY)&(intervals.reference=="ratio_50")]
    acceptance=clean_ci[clean_ci.metric=="in_scope_acceptance"]
    f1=clean_ci[clean_ci.metric=="oos_f1"]
    if (acceptance.ci_high_pp<0).all() and ((f1.ci_low_pp<=0)&(f1.ci_high_pp>=0)).all():
        clean_interpretation=("本次两个低比例候选相对50%的clean接受率差值区间均完全低于0，支持此固定设置下接受率下降；"
                              "不能把它只说成没有证据的小波动。相对50%的clean F1均值也下降，但两个区间均跨0，尚未排除无差异。"
                              "严格均值保留判断与条件区间判断分开，均不构成总体非劣效或等效结论。")
    else:
        clean_interpretation="逐项按差值与条件区间判断；均值下降和区间是否排除0是不同问题，不能把跨0解释为已证明无损。"
    validation_rows=[]
    for arm in ARMS:
        group=validation[(validation.arm==arm)&(validation.policy==FOCUS_POLICY)]
        validation_rows.append([ARM_NAMES[arm],f"{100*group.in_scope_acceptance.mean():.4f}%",f"{100*group.oos_recall.mean():.4f}%"])
    error_rows=[]
    for candidate in CANDIDATES:
        for condition in CONDITIONS[1:]:
            for metric in ("in_scope_false_rejection_rate","in_scope_correct_and_accepted_rate","in_scope_wrong_and_accepted_rate","oos_false_acceptance_rate"):
                parts=[]
                for reference in REFERENCES:
                    row=intervals[(intervals.candidate==candidate)&(intervals.reference==reference)&(intervals.condition==condition)
                                  &(intervals.policy==FOCUS_POLICY)&(intervals.metric==metric)].iloc[0]
                    parts.append(f"{row.estimate_delta_pp:+.3f} [{row.ci_low_pp:+.3f}, {row.ci_high_pp:+.3f}]")
                error_rows.append([ARM_NAMES[candidate],CONDITION_NAMES[condition],NAMES[metric],*parts])
    for row in summary.itertuples():
        values=pd.Series(row._asdict())
        full_rows.append([ARM_NAMES[row.arm],CONDITION_NAMES[row.condition],POLICY_NAMES[row.policy],*[fmt(values,m) for m in METRICS]])
    evidence_html=evidence_summary(evidence,inputs)
    evidence_html+="".join(f"<details><summary>完整证据：{html.escape(name)}</summary><pre>{html.escape(json.dumps(value,ensure_ascii=False,indent=2))}</pre></details>" for name,value in evidence.items())
    now=datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M UTC+08:00")
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>固定BERT：拒识拟合扰动比例实验</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f3f6f7;color:#193746;font:16px/1.75 system-ui,"Microsoft YaHei",sans-serif}}
main{{max-width:1280px;margin:28px auto;padding:34px 40px;background:white;border:1px solid #dbe5e8;border-radius:16px}}
h1{{font-size:29px;line-height:1.4}}h2{{font-size:23px;margin-top:34px}}.meta{{font-size:13px;color:#5c7280}}.lead{{background:#fff5e9;border-left:5px solid #c27b35;padding:16px 20px}}
.table-scroll{{overflow-x:auto;margin:16px 0}}table{{border-collapse:collapse;width:100%;font-size:12px;font-variant-numeric:tabular-nums}}
th,td{{padding:9px 10px;border-bottom:1px solid #dce5e8;text-align:left;vertical-align:top}}th{{background:#eaf0f3;white-space:nowrap}}td{{overflow-wrap:anywhere}}
img{{max-width:100%;height:auto;border:1px solid #dce5e8;border-radius:8px;margin:16px 0}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}}
details{{margin:18px 0}}summary{{cursor:pointer;font-weight:650}}@media(max-width:700px){{main{{padding:24px 16px;margin:0;border-radius:0}}}}
</style><main><p class="meta">MFE310TC · 探索性预先列明的比例实验 · {now}</p>
<h1>降低拒识拟合池中的扰动比例：收益与代价如何变化？</h1>
<p class="lead">{html.escape(' '.join(lead))}</p>
<p>完整展示0%/10%/25%/50%四档，不根据本次test挑选“最佳”比例。比例基准是固定30,000行的0%档；
历史clean_reference使用15,000行，单独作为参考，不能和30,000行0%档混称。以下主表聚焦clean validation IS接受率≥95%策略，两种规则的结果均保留。</p>
<p>降低比例可以改变误拒与错误放行之间的取舍，但不能先验称为“无损折中”。应分别对照0%检验保留多少误拒收益，
对照50%检验是否减少错误放行，再单独核对clean能力；本报告不从test选择部署比例。</p>
<h2>固定了什么，改变了什么？</h2>
<p>冻结同一批增强BERT、全部训练/验证/test缓存CLS与logits，没有新前向推理或优化。每个30k拟合池中，每条原始训练请求占两个位置：
第一份保持clean，第二份只在预先选中的source位置替换为既有noisy向量。所有比例的每源权重相同，但30k视图只有15k独立来源。</p>
{htable(['拟合臂','拟合行数','计划噪声比例','选中noisy视图','light / medium','实际改变数','改变/全部拟合位置'],data_rows)}
<p>分配种子{data['allocation_seed']}。每个intent中，light/medium两个50源组各排序一次，10%档各取前10，25%档各取前25，50%档取全部，
形成嵌套子集并在三个模型seed间共享。不能把计划比例当成实际改变覆盖；无法修改的fallback保留，未按测试表现筛源。
阈值仅用原clean validation的3,000 IS与100 OOS选择，全部15拒识器/30阈值冻结后才评价test；旧三臂直接复用。</p>
{htable(['拟合臂','95策略实际validation IS接受率（三seed均值）','validation OOS recall（三seed均值）'],validation_rows)}
<p>同一“至少95%”规则不表示各档验证接受率恰好相同；实际选择的运行点也会变化。95%约束不能保证test接受率达到95%，
不能单凭不同拟合分布下的原始阈值大小判断哪档更宽松。</p>
<p>类别中心、共享协方差、Ledoit–Wolf收缩和验证阈值共同变化。固定行数排除了单纯拟合行数变化，但不固定数据驱动的收缩系数。
马氏距离用于异常评分，分类头完全共享；各档闭集Accuracy与Macro-F1逐样本核验不变。原始距离跨不同协方差不能直接作为机制改善证明。</p>
<h2>完整比例曲线：正常误拒与错误放行必须一起看</h2>
<p>前三个IS指标以4,500条正常请求为分母；OOS错误接受以1,000条OOS为分母，等于1−OOS recall。
“正常错分类放行”表示分类头预测错误却仍被接受，不能记为恢复正确服务。百分比为三个固定seed均值±样本SD。</p>
{htable(['拟合臂','输入条件','正常误拒 %','正常正确且接受 %','正常错分类放行 %','OOS错误接受 %','OOS F1 %'],main_rows)}{image_tag(plots[0])}
<h2>10%与25%保留了多少50%档的误拒收益？</h2>
<p>收益保留=100×(mean FR<sub>0%</sub>−mean FR<sub>候选</sub>)/(mean FR<sub>0%</sub>−mean FR<sub>50%</sub>)，
只在分母大于0时定义。它是以0%为起点的误拒降低比例，不是准确率、不以历史15k clean作分母，也不是总体质量分数；不强制截断到0–100%。</p>
<p>每seed的IS分母同为4,500，代码先对整数误拒计数求和并作差，再除以13,500，避免理论为0的分母被浮点舍入误判为正；不设事后容差。</p>
{htable(['候选','条件','0%误拒','候选误拒','50%误拒','误拒收益保留'],benefit_rows)}
<h2>降低比例是否减少错误放行？</h2><p>以下差值均为候选减参照，单位pp，括号内为95%配对条件区间。
误拒、正常错分类放行与OOS错误接受为负更好；正常正确且接受为正更好。对0%和50%分别回答不同问题，不互相替代。</p>
{htable(['候选','条件','指标','候选−0%：差值 [CI]','候选−50%：差值 [CI]'],error_rows)}
<h2>clean能力：相对50%改善，不等于相对0%或历史clean保留</h2>
{htable(['拟合臂','clean OOS F1 %','clean OOS recall %','clean IS接受率 %'],clean_rows)}{image_tag(plots[1])}
<p>{html.escape('；'.join(verdicts))}。严格描述性保留要求三项均值全部不下降；不事后设margin，不把CI跨0当作非劣效或等效证明。</p>
<p>{html.escape(clean_interpretation)}</p>
{htable(['候选','参照','clean指标','差值 pp','95%条件区间 pp','该项方向','该组三项判断'],guardrail_rows)}
<h2>条件区间与固定分类头流转</h2>{image_tag(plots[2])}
<p>{repeats:,}次bootstrap、随机种子{seed}。150个intent每类有放回抽30条，OOS抽1,000条；同一原文的权重共同用于所有比例、seed、条件和策略。
每seed先算指标（包括非线性F1），再求三模型均值并作差，不把3seed输出当独立文本。区间取2.5%/97.5%分位，条件于固定模型、缓存、嵌套子集、拟合统计与阈值；
不包含新子集抽选、重拟合、重新选阈值或新噪声的不确定性，多个比较未校正。收益保留为mean比值的描述量，不另作显著性宣称。</p>
<p>rejection_transitions.csv记录0%/50%参照→10%/25%候选的受→拒、拒→受，按固定分类头正确/错误分层。
净拒绝变化=受→拒−拒→受；正确组的拒→受是恢复正确服务，错误组的拒→受是额外错误放行。normal_states.csv保留全部四状态。</p>
<h2>边界与独立证据</h2><p>本轮比例在此次评价前锁定，但由此前test发现推动，仍是重复查看测试集后的探索实验，不是全新确认集。
只测试一个嵌套来源子集实现及同一合成扰动家族，不证明自然噪声或任意子集均适用。合成文本继承原标签，未逐条人工确认语义保持。
原3处train–validation、2处train–test重叠仍存在；排除599/938的敏感性也不能消除历史曝光。</p>
<p>报告从45份预测复算90个test设置，逐条检查所有比例分类预测相同，历史clean/0%/50%分数和阈值复用一致；只写派生报告，不修改冻结模型或评价。</p>
{evidence_html}<details><summary>全部比例和两种策略结果</summary>{htable(['拟合臂','条件','策略',*[NAMES[m]+' %' for m in METRICS]],full_rows)}</details>
<details><summary>输入文件与SHA256</summary>{htable(['输入','SHA256'],[[str(p),sha256(p)] for p in inputs])}</details>
<p class="meta">产物：summary_mean_sd.csv、clean_guardrails.csv、benefit_retention.csv、normal_states.csv、rejection_transitions.csv、paired_intervals.csv及3张PNG。
AI协助实现与核验；学生应检查解释并按课程要求披露实际AI用途和个人贡献。</p></main></html>'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=PROJECT/"outputs")
    parser.add_argument("--bootstrap-repeats",type=int,default=2000)
    parser.add_argument("--bootstrap-seed",type=int,default=20261021)
    parser.add_argument("--batch-size",type=int,default=64)
    args=parser.parse_args();output=args.output.resolve()
    metrics,frames,inputs=load_results(output)
    validation_path=output/"validation_metrics_by_seed.csv"
    validation=pd.read_csv(validation_path,float_precision="round_trip")
    if len(validation)!=30:raise ValueError("Expected 30 validation settings")
    inputs.append(validation_path)
    data_path=PROJECT/"data/data_summary.json";data=json.loads(data_path.read_text(encoding="utf-8"));inputs.append(data_path)
    evidence={}
    for path in (PROJECT/"data_audit.json",output/"verification.json",output/"duplicate_sensitivity.json",output/"fixed_head_rejection_diagnostic.json"):
        if path.is_file():evidence[path.name]=json.loads(path.read_text(encoding="utf-8"));inputs.append(path)
    summary=aggregate(metrics,["arm","condition","score","policy"])
    guardrails,retention=derived_tables(summary,metrics)
    states,transitions=states_and_transitions(frames)
    intervals=paired_bootstrap(frames,args.bootstrap_repeats,args.bootstrap_seed,args.batch_size)
    plots=create_plots(summary,retention,intervals,output)
    document=build_html(summary,guardrails,retention,intervals,plots,data,inputs,evidence,args.bootstrap_repeats,args.bootstrap_seed,validation)
    for name,frame in (("summary_mean_sd.csv",summary),("clean_guardrails.csv",guardrails),("benefit_retention.csv",retention),
                       ("normal_states.csv",states),("rejection_transitions.csv",transitions),("paired_intervals.csv",intervals)):
        frame.to_csv(output/name,index=False)
    report=output/"ratio_report_cn.html";report.write_text(document,encoding="utf-8")
    print(json.dumps(dict(status="PASS",verified_settings=len(metrics),normal_states=len(states),rejection_transitions=len(transitions),
                          clean_guardrails=len(guardrails),benefit_retention=len(retention),paired_intervals=len(intervals),report=str(report)),ensure_ascii=False))


if __name__=="__main__":main()
