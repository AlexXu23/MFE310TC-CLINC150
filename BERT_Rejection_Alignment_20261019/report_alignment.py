"""Report fixed-score IS-only validation-acceptance alignment.

No inference, fitting or threshold selection. Run after TEST_COMPLETE.json.
Gap change is old signed gap minus aligned signed gap, not a causal percentage.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timedelta,timezone
import html
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

PROJECT=Path(__file__).resolve().parent
PRIOR=PROJECT.parent/"BERT_Rejection_Ratio_20261019"
sys.path.insert(0,str(PRIOR))
from report_ratio import ARMS,GRID,CANDIDATES,REFERENCES,ARM_NAMES,METRICS,NAMES,SEEDS,CONDITIONS,CONDITION_NAMES
from report_ratio import booleans,evaluate_predictions,htable,aggregate,sha256,manifest_hashes,counts_to_metrics

OLD_POLICY="is_acceptance_95"
NEW_POLICY="aligned_is_acceptance_95"
POLICIES=(OLD_POLICY,NEW_POLICY)
POLICY_NAMES={OLD_POLICY:"旧≥95%且使用OOS选择",NEW_POLICY:"IS-only共同接受数对齐"}
STATES=("accepted_correct","accepted_wrong","rejected_correct","rejected_wrong")


def load_results(output):
    manifests=[PROJECT/"INPUTS_FROZEN.json",output/"DEVELOPMENT_FROZEN.json",output/"TEST_COMPLETE.json"]
    hashes={}
    for path in manifests:hashes.update(manifest_hashes(json.loads(path.read_text(encoding="utf-8"))))
    metric_path=output/"metrics_by_seed.csv"
    metrics=pd.read_csv(metric_path,float_precision="round_trip")
    keys=["arm","seed","condition","score","policy"]
    if len(metrics)!=90 or metrics.duplicated(keys).any():raise ValueError("Expected 90 test settings")
    for name,values in (("arm",ARMS),("seed",SEEDS),("condition",CONDITIONS),("score",("mahalanobis",)),("policy",POLICIES)):
        if set(metrics[name])!=set(values):raise ValueError(f"Unexpected {name}")
    history_path=PRIOR/"outputs/metrics_by_seed.csv"
    history=pd.read_csv(history_path,float_precision="round_trip")
    history=history[history.policy.eq(OLD_POLICY)].set_index(["arm","seed","condition"])
    inputs=[metric_path,history_path,*manifests]
    identity=["row_id","original_text","true_label","label_id","is_oos"]
    shared_cols=identity+["text","changed","modified_tokens","predicted_label_id"]
    frames,canonical,shared={},None,{}
    for arm in ARMS:
        frames[arm]={}
        for seed in SEEDS:
            frames[arm][seed]={}
            for condition in CONDITIONS:
                path=output/arm/f"seed{seed}"/f"{condition}_predictions.csv"
                prior_path=PRIOR/"outputs"/arm/f"seed{seed}"/f"{condition}_predictions.csv"
                frame=pd.read_csv(path,float_precision="round_trip")
                prior=pd.read_csv(prior_path,float_precision="round_trip")
                required=set(shared_cols+["mahalanobis_score"]+[f"mahalanobis_{p}_reject" for p in POLICIES])
                if not required.issubset(frame) or frame[list(required)].isna().any().any() or frame.row_id.duplicated().any():
                    raise ValueError(f"Invalid predictions: {path}")
                # The runner copies every existing field; no cached result may drift.
                if not set(prior).issubset(frame) or not frame[prior.columns].equals(prior):
                    raise ValueError("Existing prediction columns changed during rethresholding")
                frame["is_oos"],frame["changed"]=booleans(frame.is_oos),booleans(frame.changed)
                if canonical is None:canonical=frame[identity].copy()
                else:
                    if set(frame.row_id)!=set(canonical.row_id):raise ValueError("Row identities differ")
                    frame=frame.set_index("row_id").loc[canonical.row_id].reset_index()
                    if not frame[identity].equals(canonical):raise ValueError("Original labels/text differ")
                if arm==ARMS[0]:shared[(seed,condition)]=frame[shared_cols].copy()
                elif not frame[shared_cols].equals(shared[(seed,condition)]):raise ValueError("Classifier predictions differ across arms")
                labels,oos,prediction=frame.label_id.to_numpy(),frame.is_oos.to_numpy(),frame.predicted_label_id.to_numpy()
                unique,counts=np.unique(labels[~oos],return_counts=True)
                if (len(frame)!=5500 or oos.sum()!=1000 or not np.all(labels[oos]==-1)
                        or not np.array_equal(unique,np.arange(150)) or not np.all(counts==30)):
                    raise ValueError("Expected 150x30 IS and 1000 OOS")
                values=frame.mahalanobis_score.to_numpy(dtype=float)
                if not np.isfinite(values).all():raise ValueError("Invalid scores")
                closed=float((prediction[~oos]==labels[~oos]).mean())
                macro=float(f1_score(labels[~oos],prediction[~oos],labels=np.arange(150),average="macro"))
                cutoffs={}
                for policy in POLICIES:
                    mask=metrics.arm.eq(arm)&metrics.seed.eq(seed)&metrics.condition.eq(condition)&metrics.policy.eq(policy)
                    row=metrics.loc[mask].iloc[0]
                    frozen=metrics[metrics.arm.eq(arm)&metrics.seed.eq(seed)&metrics.policy.eq(policy)]
                    if frozen.threshold.nunique(dropna=False)!=1 or pd.isna(row.threshold):raise ValueError("Test threshold drift")
                    cutoffs[policy]=float(row.threshold)
                    rejected=booleans(frame[f"mahalanobis_{policy}_reject"])
                    frame[f"mahalanobis_{policy}_reject"]=rejected
                    if not np.array_equal(rejected,values>=float(row.threshold)):raise ValueError("Threshold/decision mismatch")
                    calculated=evaluate_predictions(labels,prediction,oos,values,rejected)
                    calculated.update(closed_set_accuracy=closed,closed_set_macro_f1=macro)
                    for metric,value in calculated.items():
                        if not np.isclose(value,float(row[metric]),rtol=1e-8,atol=1e-10,equal_nan=True):
                            raise ValueError(f"Metric mismatch: {arm}/{seed}/{condition}/{policy}/{metric}")
                    metrics.loc[mask,"in_scope_wrong_and_accepted_rate"]=float((~oos&~rejected&(prediction!=labels)).sum()/4500)
                    metrics.loc[mask,"oos_false_acceptance_rate"]=float((oos&~rejected).sum()/1000)
                    if policy==OLD_POLICY:
                        old=history.loc[(arm,seed,condition)]
                        if float(old.threshold)!=float(row.threshold):raise ValueError("Old threshold changed")
                        for metric in calculated:
                            if not np.isclose(float(old[metric]),float(row[metric]),rtol=1e-8,atol=1e-10,equal_nan=True):
                                raise ValueError("Old-policy evaluation changed")
                before=frame[f"mahalanobis_{OLD_POLICY}_reject"].to_numpy()
                after=frame[f"mahalanobis_{NEW_POLICY}_reject"].to_numpy()
                if cutoffs[NEW_POLICY]<=cutoffs[OLD_POLICY] and np.any(before&~after):raise ValueError("Lower threshold removed rejections")
                if cutoffs[NEW_POLICY]>=cutoffs[OLD_POLICY] and np.any(~before&after):raise ValueError("Higher threshold added rejections")
                frames[arm][seed][condition]=frame;inputs.extend([path,prior_path])
    for _,group in metrics.groupby(["seed","condition"]):
        if group.closed_set_accuracy.nunique()!=1 or group.closed_set_macro_f1.nunique()!=1:raise ValueError("Closed metrics changed")
    for path in inputs:
        expected=hashes.get(str(path.resolve()))
        if expected and sha256(path)!=expected:raise ValueError(f"Frozen consumed file changed: {path}")
    alignment_path=output/"alignment_summary.json"
    points_path=output/"operating_points.csv"
    validation_path=output/"validation_metrics_by_seed.csv"
    alignment=json.loads(alignment_path.read_text(encoding="utf-8"))
    points=pd.read_csv(points_path,float_precision="round_trip")
    validation=pd.read_csv(validation_path,float_precision="round_trip")
    if len(points)!=15 or len(validation)!=30:raise ValueError("Invalid alignment metadata size")
    if alignment["selection_used_oos"] or not np.all(points.actual_accepted==alignment["actual_common_accepted"]):
        raise ValueError("Alignment did not use one IS-only common count")
    if not np.allclose(validation.loc[validation.policy.eq(NEW_POLICY),"in_scope_acceptance"],alignment["actual_common_acceptance"],rtol=0,atol=1e-14):
        raise ValueError("Achieved validation acceptance differs across aligned detectors")
    inputs.extend([alignment_path,points_path,validation_path])
    return metrics,frames,inputs,alignment,points,validation


def metric_changes(metrics):
    index=metrics.set_index(["arm","seed","condition","policy"])
    within,gaps=[],[]
    for arm in ARMS:
        for seed in SEEDS:
            for condition in CONDITIONS:
                for metric in METRICS:
                    old,new=(float(index.loc[(arm,seed,condition,p)][metric]) for p in POLICIES)
                    within.append(dict(arm=arm,seed=seed,condition=condition,score="mahalanobis",metric=metric,
                                       old_value=old,aligned_value=new,aligned_minus_original=new-old,aligned_minus_original_pp=100*(new-old)))
    for candidate in CANDIDATES:
        for reference in REFERENCES:
            for seed in SEEDS:
                for condition in CONDITIONS:
                    for metric in METRICS:
                        old,new=(float(index.loc[(candidate,seed,condition,p)][metric])-float(index.loc[(reference,seed,condition,p)][metric]) for p in POLICIES)
                        gaps.append(dict(candidate=candidate,reference=reference,comparison=candidate+"_minus_"+reference,
                                         seed=seed,condition=condition,score="mahalanobis",metric=metric,old_gap=old,aligned_gap=new,gap_change=old-new,
                                         old_gap_pp=100*old,aligned_gap_pp=100*new,gap_change_pp=100*(old-new)))
    return pd.DataFrame(within),pd.DataFrame(gaps)


def states_and_transitions(frames):
    states,transitions=[],[]
    for arm in ARMS:
        for seed in SEEDS:
            for condition in CONDITIONS:
                frame=frames[arm][seed][condition]
                oos=frame.is_oos.to_numpy();correct=frame.predicted_label_id.to_numpy()==frame.label_id.to_numpy()
                before=frame[f"mahalanobis_{OLD_POLICY}_reject"].to_numpy()
                after=frame[f"mahalanobis_{NEW_POLICY}_reject"].to_numpy()
                for policy,rejected in zip(POLICIES,(before,after)):
                    row=dict(arm=arm,seed=seed,condition=condition,score="mahalanobis",policy=policy,n_is=4500)
                    for name,event in zip(STATES,(~rejected&correct,~rejected&~correct,rejected&correct,rejected&~correct)):
                        row[name+"_n"]=int((event&~oos).sum());row[name+"_rate_over_all_is"]=row[name+"_n"]/4500
                    states.append(row)
                for stratum,mask,scope_n in (("all_is",~oos,4500),("head_correct",~oos&correct,4500),("head_wrong",~oos&~correct,4500),("oos",oos,1000)):
                    n=int(mask.sum());row=dict(arm=arm,seed=seed,condition=condition,score="mahalanobis",stratum=stratum,
                                               old_policy=OLD_POLICY,new_policy=NEW_POLICY,group_n=n,scope_n=scope_n)
                    for name,event in (("accept_accept",~before&~after),("accept_reject",~before&after),("reject_accept",before&~after),("reject_reject",before&after)):
                        row[name+"_n"]=int((mask&event).sum());row[name+"_rate_over_group"]=row[name+"_n"]/n if n else np.nan
                        row[name+"_rate_over_scope"]=row[name+"_n"]/scope_n
                    row["net_rejection_change_n"]=row["accept_reject_n"]-row["reject_accept_n"]
                    transitions.append(row)
    return pd.DataFrame(states),pd.DataFrame(transitions)


def paired_bootstrap(frames,repeats=2000,random_seed=20261022,batch_size=64):
    if repeats<2 or batch_size<1:raise ValueError("Invalid bootstrap configuration")
    first=frames[ARMS[0]][SEEDS[0]]["clean"]
    labels,oos=first.label_id.to_numpy(),first.is_oos.to_numpy()
    strata=[np.flatnonzero(labels==label) for label in np.unique(labels)]
    events=np.zeros((5,3,3,2,4,5500),dtype=float)
    for ai,arm in enumerate(ARMS):
        for si,seed in enumerate(SEEDS):
            for ci,condition in enumerate(CONDITIONS):
                frame=frames[arm][seed][condition];correct=(frame.predicted_label_id.to_numpy()==labels)&~oos
                for pi,policy in enumerate(POLICIES):
                    rejected=frame[f"mahalanobis_{policy}_reject"].to_numpy()
                    events[ai,si,ci,pi]=np.stack((rejected&oos,rejected&~oos,~rejected&correct,~rejected&~correct&~oos))
    def linear_numerators(counts):
        tp,fp,ac,aw=(counts[...,i,:] for i in range(4))
        return np.stack((fp,ac,aw,1000-tp,np.zeros_like(tp),tp,4500-fp,tp+ac),axis=-2).sum(axis=1)
    raw_point=events.sum(axis=-1)[...,None]
    point=counts_to_metrics(raw_point).mean(axis=1)[...,0]
    point_numerators=linear_numerators(raw_point)[...,0]
    denominator=np.array([13500,13500,13500,3000,np.nan,3000,13500,16500])
    draws=np.empty((*point.shape,repeats));draw_numerators=np.empty_like(draws);rng=np.random.default_rng(random_seed)
    for start in range(0,repeats,batch_size):
        size=min(batch_size,repeats-start)
        sample=np.concatenate([rng.choice(indices,size=(size,len(indices)),replace=True) for indices in strata],axis=1)
        weights=np.bincount((sample+5500*np.arange(size)[:,None]).ravel(),minlength=size*5500).reshape(size,5500).astype(float)
        counts=(events.reshape(-1,5500)@weights.T).reshape(*events.shape[:-1],size)
        draws[...,start:start+size]=counts_to_metrics(counts).mean(axis=1)
        draw_numerators[...,start:start+size]=linear_numerators(counts)
    rows=[]
    def add(base,estimate,distribution):
        low,high=np.quantile(distribution,[.025,.975])
        rows.append(dict(**base,estimate=estimate,ci_low=low,ci_high=high,estimate_pp=100*estimate,ci_low_pp=100*low,ci_high_pp=100*high,
                         bootstrap_repeats=repeats,bootstrap_seed=random_seed,n_is=4500,n_oos=1000,n_fixed_models=3,
                         scope="fixed classifier, scores and both threshold sets; shared text draws across arms/policies/seeds/conditions"))
    for ai,arm in enumerate(ARMS):
        for ci,condition in enumerate(CONDITIONS):
            for mi,metric in enumerate(METRICS):
                estimate=point[ai,ci,1,mi]-point[ai,ci,0,mi]
                distribution=draws[ai,ci,1,mi]-draws[ai,ci,0,mi]
                if metric!="oos_f1":
                    estimate=(point_numerators[ai,ci,1,mi]-point_numerators[ai,ci,0,mi])/denominator[mi]
                    distribution=(draw_numerators[ai,ci,1,mi]-draw_numerators[ai,ci,0,mi])/denominator[mi]
                add(dict(contrast_type="within_arm",arm=arm,candidate="",reference="",condition=condition,score="mahalanobis",metric=metric,estimand="aligned_minus_original"),
                    estimate,distribution)
    for candidate in CANDIDATES:
        for reference in REFERENCES:
            a,b=ARMS.index(candidate),ARMS.index(reference)
            for ci,condition in enumerate(CONDITIONS):
                for mi,metric in enumerate(METRICS):
                    old,new=point[a,ci,0,mi]-point[b,ci,0,mi],point[a,ci,1,mi]-point[b,ci,1,mi]
                    old_draw,new_draw=draws[a,ci,0,mi]-draws[b,ci,0,mi],draws[a,ci,1,mi]-draws[b,ci,1,mi]
                    change,change_draw=old-new,old_draw-new_draw
                    if metric!="oos_f1":
                        old_num=point_numerators[a,ci,0,mi]-point_numerators[b,ci,0,mi]
                        new_num=point_numerators[a,ci,1,mi]-point_numerators[b,ci,1,mi]
                        old_draw_num=draw_numerators[a,ci,0,mi]-draw_numerators[b,ci,0,mi]
                        new_draw_num=draw_numerators[a,ci,1,mi]-draw_numerators[b,ci,1,mi]
                        old,new=old_num/denominator[mi],new_num/denominator[mi]
                        old_draw,new_draw=old_draw_num/denominator[mi],new_draw_num/denominator[mi]
                        change=(old_num-new_num)/denominator[mi]
                        change_draw=(old_draw_num-new_draw_num)/denominator[mi]
                    base=dict(contrast_type="cross_arm",arm="",candidate=candidate,reference=reference,condition=condition,score="mahalanobis",metric=metric)
                    for name,value,dist in (("old_gap",old,old_draw),("aligned_gap",new,new_draw),("gap_change",change,change_draw)):
                        add(dict(**base,estimand=name),value,dist)
    return pd.DataFrame(rows)


def create_plots(summary,intervals,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    table=summary.set_index(["arm","condition","score","policy"])
    x=np.array([0,10,25,50]);colors=("#397c91","#c27b35","#6e669f")
    panels=(("in_scope_false_rejection_rate","IS false rejection (%)"),("in_scope_correct_and_accepted_rate","IS correct + accepted (%)"),
            ("in_scope_wrong_and_accepted_rate","Wrong-head accepted / all IS (%)"),("oos_false_acceptance_rate","OOS false acceptance (%)"))
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for ax,(metric,title) in zip(axes.flat,panels):
        for color,condition in zip(colors,CONDITIONS):
            for policy,style,marker in ((OLD_POLICY,"--",None),(NEW_POLICY,"-","o")):
                means=[100*table.loc[(arm,condition,"mahalanobis",policy)][metric+"_mean"] for arm in GRID]
                ax.plot(x,means,style,marker=marker,color=color,label=condition.replace("typo_","").title() if policy==NEW_POLICY else None)
        ax.set_xticks(x);ax.set_xlabel("Planned noisy ratio in fixed 30k fit pool (%)");ax.set_ylabel(title)
        ax.spines[["top","right"]].set_visible(False)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc="outside lower center",ncol=3)
    fig.suptitle("Fixed scores: dashed = original rule; solid = common IS-only validation acceptance | 3-seed means")
    curves=output/"alignment_ratio_curves.png";fig.savefig(curves,dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4.8),constrained_layout=True)
    for ax,metric in zip(axes,("oos_f1","oos_recall","in_scope_acceptance")):
        for policy,color,style in ((OLD_POLICY,colors[1],"--"),(NEW_POLICY,colors[0],"-")):
            ax.plot(x,[100*table.loc[(arm,"clean","mahalanobis",policy)][metric+"_mean"] for arm in GRID],style,marker="o",color=color,label="Original" if policy==OLD_POLICY else "Aligned")
        ax.set_xticks(x);ax.set_xlabel("Planned noisy ratio (%)");ax.set_ylabel("Clean test (%)");ax.set_title(metric.replace("_"," "))
        ax.spines[["top","right"]].set_visible(False)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="outside lower center",ncol=2)
    fig.suptitle("Common validation acceptance does not force common test acceptance | 3-seed means")
    clean=output/"alignment_clean_comparison.png";fig.savefig(clean,dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(14,10),constrained_layout=True)
    selected_endpoints=(("clean","in_scope_acceptance"),("clean","oos_f1"),("typo_light","in_scope_false_rejection_rate"),
                        ("typo_medium","in_scope_false_rejection_rate"),("typo_light","in_scope_wrong_and_accepted_rate"),
                        ("typo_medium","in_scope_wrong_and_accepted_rate"),("typo_light","oos_false_acceptance_rate"),("typo_medium","oos_false_acceptance_rate"))
    endpoint_names={"in_scope_acceptance":"IS acceptance","oos_f1":"OOS F1","in_scope_false_rejection_rate":"IS false rejection", "in_scope_wrong_and_accepted_rate":"Wrong IS accepted","oos_false_acceptance_rate":"OOS false accept"}
    for ax,(candidate,reference) in zip(axes.flat,[(c,r) for c in CANDIDATES for r in REFERENCES]):
        for estimand,color,offset in (("old_gap",colors[1],-.13),("aligned_gap",colors[0],.13)):
            rowset=intervals[(intervals.contrast_type=="cross_arm")&(intervals.candidate==candidate)&(intervals.reference==reference)&(intervals.estimand==estimand)].set_index(["condition","metric"])
            rows=rowset.loc[list(selected_endpoints)]
            y=np.arange(len(rows))+offset
            ax.hlines(y,rows.ci_low_pp,rows.ci_high_pp,color=color,lw=2)
            ax.scatter(rows.estimate_pp,y,color=color,s=20,label="Old gap" if estimand=="old_gap" else "Aligned gap")
        labels=[c.replace("typo_","").title()+": "+endpoint_names[m] for c,m in selected_endpoints]
        ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis();ax.axvline(0,color="#8a989f",ls="--",lw=1)
        ax.set_title(candidate[-2:]+"% minus "+reference[-2:]+"%");ax.set_xlabel("Signed candidate minus reference gap (pp)")
        ax.spines[["top","right"]].set_visible(False)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc="outside lower center",ncol=2)
    fig.suptitle("Old and aligned cross-ratio gaps | paired 95% conditional intervals")
    gap=output/"alignment_gap_intervals.png";fig.savefig(gap,dpi=180);plt.close(fig)
    return curves,clean,gap


def evidence_html(evidence,inputs):
    parts=[]
    verification=evidence.get("verification.json")
    if verification:
        if verification.get("status")!="PASS":raise ValueError("Independent verification failed")
        if verification.get("metrics_sha256") and verification["metrics_sha256"]!=sha256(inputs[0]):raise ValueError("Verification metrics hash differs")
        check=verification["alignment_check"]
        fields=[["test / validation设置",f"{verification['verified_test_settings']} / {verification['verified_validation_settings']}"],
                ["IS-only阈值与共同接受数",f"{check['models']}组，{check['actual_common_is_accepted']}/3000"],
                ["原列、原分数、旧政策与分类头","逐条一致"],
                ["派生表核验",verification["report_checks"]["status"]]]
        parts.append("<p>独立验证状态：PASS。阈值选择和拒绝集合单调性均独立核验；完整记录与哈希见<a href=\"verification.json\">verification.json</a>。</p>"+htable(["验证项目","结果"],fields))
    audit=evidence.get("threshold_audit.json")
    if audit:
        parts.append(f"<p>第二份独立阈值审计复算{audit['candidate_rows_checked']:,}条IS候选，确认共同可行接受数和最小候选规则；"
                     '<a href="threshold_audit.json">原始阈值审计</a>保留全部逐模型证据。</p>')
    sensitivity=evidence.get("duplicate_sensitivity.json")
    if sensitivity:
        maximum=sensitivity["max_absolute_exclusion_change_in_method_gap_pp"]
        gapmaximum=sensitivity["max_absolute_exclusion_change_in_gap_decomposition_pp"]
        parts.append(f"<p>全部臂和政策统一排除原重复row_id 599/938后，方法gap最大变化{maximum:.8f} pp，"
                     f"gap分解变化最大{gapmaximum:.8f} pp。只重算固定结果，不能撤销历史训练/验证或test曝光。</p>")
        parts.append('<p>完整敏感性记录：<a href="duplicate_sensitivity.json">duplicate_sensitivity.json</a>。</p>')
    return "".join(parts)


def format_ci(row):
    return f"{row.estimate_pp:+.4f} [{row.ci_low_pp:+.4f}, {row.ci_high_pp:+.4f}]"


def build_html(summary,within,gaps,intervals,plots,alignment,points,validation,inputs,evidence,repeats,random_seed):
    table=summary.set_index(["arm","condition","score","policy"])
    ci=intervals[intervals.contrast_type.eq("cross_arm")].set_index(["candidate","reference","condition","metric","estimand"])
    within_ci=intervals[intervals.contrast_type.eq("within_arm")].set_index(["arm","condition","metric"])
    point_rows=[]
    for row in points.itertuples():
        point_rows.append([ARM_NAMES[row.arm],row.seed,int(row.original_accepted),int(row.actual_accepted),f"{row.original_threshold:.7g}",f"{row.aligned_threshold:.7g}"])
    aligned_rows,clean_rows,changes=[],[],[]
    for arm in ARMS:
        for condition in CONDITIONS:
            row=table.loc[(arm,condition,"mahalanobis",NEW_POLICY)]
            aligned_rows.append([ARM_NAMES[arm],CONDITION_NAMES[condition],*[f"{100*row[m+'_mean']:.3f} ± {100*row[m+'_sd']:.3f}" for m in METRICS[:4]],f"{100*row.oos_f1_mean:.3f}"])
            changes.append([ARM_NAMES[arm],CONDITION_NAMES[condition],*[format_ci(within_ci.loc[(arm,condition,m)]) for m in METRICS[:4]]])
        for policy in POLICIES:
            row=table.loc[(arm,"clean","mahalanobis",policy)]
            clean_rows.append([ARM_NAMES[arm],"原规则" if policy==OLD_POLICY else "对齐后",*[f"{100*row[m+'_mean']:.4f}%" for m in ("oos_f1","oos_recall","in_scope_acceptance")]])
    gap_rows=[];lead=[]
    chosen=(("clean","in_scope_acceptance"),("clean","oos_f1"),("clean","oos_recall"),
            ("typo_light","in_scope_false_rejection_rate"),("typo_medium","in_scope_false_rejection_rate"),
            ("typo_light","in_scope_wrong_and_accepted_rate"),("typo_medium","in_scope_wrong_and_accepted_rate"),
            ("typo_light","oos_false_acceptance_rate"),("typo_medium","oos_false_acceptance_rate"))
    for candidate in CANDIDATES:
        for reference in REFERENCES:
            for condition,metric in chosen:
                rows=[ci.loc[(candidate,reference,condition,metric,estimand)] for estimand in ("old_gap","aligned_gap","gap_change")]
                gap_rows.append([ARM_NAMES[candidate]+" − "+ARM_NAMES[reference],CONDITION_NAMES[condition],NAMES[metric],*[format_ci(r) for r in rows]])
                if reference=="ratio_50" and condition=="clean" and metric=="in_scope_acceptance":
                    lead.append(f"{candidate[-2:]}%相对50%的clean接受率差：原{rows[0].estimate_pp:+.3f} pp → 对齐后{rows[1].estimate_pp:+.3f} pp；旧gap−新gap={rows[2].estimate_pp:+.3f} pp，95%条件区间[{rows[2].ci_low_pp:+.3f}, {rows[2].ci_high_pp:+.3f}]。")
    example_rows=[]
    for condition in CONDITIONS[1:]:
        old,new,change=(ci.loc[("ratio_25","ratio_50",condition,"in_scope_false_rejection_rate",e)] for e in ("old_gap","aligned_gap","gap_change"))
        example_rows.append([CONDITION_NAMES[condition],f"{old.estimate_pp:+.4f}",format_ci(new),format_ci(change)])
    clean_fr=ci.loc[("ratio_25","ratio_50","clean","in_scope_false_rejection_rate","aligned_gap")]
    clean_f1=ci.loc[("ratio_25","ratio_50","clean","oos_f1","aligned_gap")]
    aligned_acceptance=[table.loc[(a,"clean","mahalanobis",NEW_POLICY)].in_scope_acceptance_mean*100 for a in ARMS]
    interpretation=("在这个共同验证运行点，原先25%对50%的noisy误拒差距缩小，但对齐后仍为正，不能说差距全部消失。"
                    f"clean的25%对50%误拒差值变为{clean_fr.estimate_pp:+.4f} pp，区间[{clean_fr.ci_low_pp:+.4f}, {clean_fr.ci_high_pp:+.4f}]；"
                    f"F1差值变为{clean_f1.estimate_pp:+.4f} pp，区间[{clean_f1.ci_low_pp:+.4f}, {clean_f1.ci_high_pp:+.4f}]。"
                    "均值方向可以改变；误拒区间接触0、F1区间跨0，不据此宣称严格优于或无损保留。")
    generated=datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M UTC+08:00")
    image=lambda path:f'<img src="{html.escape(path.name,quote=True)}" alt="{html.escape(path.stem)}">'
    source_links=htable(["证据","用途"],[["<a href=\"alignment_summary.json\">alignment_summary.json</a>","共同可行接受数与逐模型选择"],["<a href=\"operating_points.csv\">operating_points.csv</a>","旧新阈值与实现接受数"]])
    # htable escapes text; use explicit local links for readable source access.
    source_links='<p><a href="alignment_summary.json">共同接受数记录</a> · <a href="operating_points.csv">逐模型运行点</a> · <a href="../PROTOCOL.json">冻结协议</a> · <a href="paired_intervals.csv">全部配对区间</a> · <a href="cross_arm_gaps.csv">逐seed差异分解</a></p>'
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>固定分数的IS接受率对齐</title>
<style>*{{box-sizing:border-box}}body{{margin:0;background:#f3f6f7;color:#193746;font:16px/1.75 system-ui,"Microsoft YaHei",sans-serif}}
main{{max-width:1300px;margin:28px auto;padding:34px 40px;background:#fff;border:1px solid #dbe5e8;border-radius:16px}}h1{{font-size:29px;line-height:1.4}}h2{{font-size:23px;margin-top:32px}}
.meta{{font-size:13px;color:#5c7280}}.lead{{background:#fff5e9;border-left:5px solid #c27b35;padding:16px 20px}}.table-scroll{{overflow-x:auto;margin:16px 0}}
table{{border-collapse:collapse;width:100%;font-size:12px;font-variant-numeric:tabular-nums}}th,td{{padding:8px 9px;border-bottom:1px solid #dce5e8;text-align:left;vertical-align:top}}
th{{background:#eaf0f3;white-space:nowrap}}td{{overflow-wrap:anywhere}}img{{max-width:100%;height:auto;border:1px solid #dce5e8;border-radius:8px;margin:16px 0}}summary{{cursor:pointer;font-weight:650}}
@media(max-width:700px){{main{{padding:24px 16px;margin:0;border-radius:0}}}}</style><main>
<p class="meta">MFE310TC · 固定分数、共同验证运行点 · {generated}</p><h1>把验证IS接受率对齐后，比例间差异改变多少？</h1>
<p class="lead">{html.escape(' '.join(lead))}</p>
<p>本轮只移动阈值。BERT、缓存向量、15个拒识分布、每条样本的距离分数与分类预测均保持不变。
它回答选定共同验证接受数下的旧新差异，不把gap变化解释为普遍因果归因百分比，也不据test挑选最优比例。</p>
<p>{html.escape(interpretation)}</p>
{htable(['25% − 50%','原误拒gap pp','对齐后误拒gap [CI] pp','原−新gap [CI] pp'],example_rows)}
<h2>IS-only对齐了哪个运行点？</h2>
<p>目标为{alignment['target_accepted']}/{alignment['validation_is_rows']}={100*alignment['target_acceptance']:.2f}%。
实际共同可行接受数为<strong>{alignment['actual_common_accepted']}/{alignment['validation_is_rows']}={100*alignment['actual_common_acceptance']:.4f}%</strong>。
只把clean validation的3,000条IS分数送入选择函数；先确定全部阈值，之后才评价100条validation OOS。</p>
<p>候选为唯一IS分数及最大分数之后的nextafter值；接受当且仅当score&lt;threshold。
先求全部15模型可实现整数接受数的交集，再取不低于2,850的最小共同数，各模型选能实现该数的最小候选阈值。
若ties阻止恰好95%，按预先规则共同上移，不随机拆tie或依据test改容差。</p>
<details><summary>15组旧新阈值和实际验证接受数</summary>{htable(['拟合臂','seed','原接受数/3000','新接受数/3000','原阈值','新阈值'],point_rows)}</details>
<p>旧规则是在IS接受率≥95%约束内使用验证OOS表现选择；新规则只用IS共同接受数。因此干预同时包含“共同运行点对齐”和“从OOS-aware转为IS-only选择”。
同一模型内阈值下移只能增加拒绝，上移只能减少拒绝；不同拟合分布之间不能按原始阈值大小判断宽松程度。clean test接受率不保证等于95%。</p>
<p>本次对齐后各臂clean test接受率均值仍只有{min(aligned_acceptance):.4f}%–{max(aligned_acceptance):.4f}%，验证集的共同95%没有自动迁移为测试集95%。</p>
<h2>完整比例曲线与对齐后表现</h2>{image(plots[0])}
<p>虚线为旧规则，实线为新规则；固定30k比例为0/10/25/50%，15k历史clean单列参考。表内IS误拒、正确接受和错分类放行均以4,500条IS为分母，
OOS错误接受以1,000条OOS为分母（1−recall）。前四项为三seed均值±样本SD，F1列为三seed均值。</p>
{htable(['拟合臂','条件','IS误拒 %','IS正确且接受 %','IS错分类放行 %','OOS错误接受 %','OOS F1 %'],aligned_rows)}
<h2>每个拒识器自身：旧→新政策改变了什么？</h2>
<p>差值为aligned−original，单位pp；方括号是95%条件区间。误拒与错误放行为负更好，正确接受为正更好。
阈值变严格可以减少错放，同时拒绝更多本可正确回答的IS，不能只看其中一项。</p>
{htable(['拟合臂','条件','IS误拒变化 [CI]','正确接受变化 [CI]','IS错放变化 [CI]','OOS错放变化 [CI]'],changes)}
<h2>clean能力：共同validation接受率不保证共同test接受率</h2>
{htable(['拟合臂','政策','clean OOS F1','clean OOS recall','clean IS接受率'],clean_rows)}{image(plots[1])}
<h2>跨比例差异：旧gap、新gap、gap变化</h2>
<p><code>old_gap = candidate_old − reference_old</code>；<code>aligned_gap = candidate_aligned − reference_aligned</code>；
<code>gap_change = old_gap − aligned_gap</code>。最后一项为有符号差值的变化，不自动等于绝对差异缩小：要同时看旧新方向、是否换号和离0的距离。
本报告只给绝对pp与配对区间，不计算可能因零分母或换号失真的“归因百分比”。</p>
{image(plots[2])}
{htable(['候选 − 参照','条件','指标','旧gap [CI] pp','新gap [CI] pp','旧−新gap [CI] pp'],gap_rows)}
<h2>如何理解这些条件区间？</h2>
<p>{repeats:,}次bootstrap，随机种子{random_seed}；150个IS intent各有放回抽30条，OOS抽1,000条。
同一原文的抽样权重共同用于所有比例、旧新政策、seed与输入条件。每seed先计算F1等指标，再对三个固定模型取均值。
gap_change在每次同一抽样中直接做差，不能把两套CI端点相减。区间取2.5%/97.5%分位，不将3seed预测视为独立文本。</p>
<p>区间条件于固定模型、分数与两套阈值，未包括重新选validation阈值、重新拟合、抽选新子集或生成新噪声的不确定性，多项比较未校正。
对齐后的残差仍可能涉及验证到测试的迁移和分数排序/分布差异；旧新gap差不是不随目标接受率变化的因果百分比。</p>
<h2>核验与复现</h2>
<p>报告逐条核对原预测CSV所有旧列不变，重算90个test设置，复现旧政策，检查分类头完全相同和阈值方向与拒绝集合的单调关系。
这是此前测试结果推动的探索性后续；既有3处train–validation和2处train–test重叠仍在，不能称为全新未曝光确认集。</p>
{evidence_html(evidence,inputs)}{source_links}
<details><summary>冻结数据与分析证据的SHA256（可更新verification仅链接）</summary>{htable(['文件','SHA256'],[[str(p),sha256(p)] for p in inputs if p.name!='verification.json'])}</details>
<p class="meta">图片使用同目录PNG，分享时保留outputs文件夹。派生产物：summary_mean_sd.csv、within_arm_changes.csv、cross_arm_gaps.csv、normal_states.csv、policy_transitions.csv、paired_intervals.csv。
AI协助实现与统计核验；学生应核验解释并按课程要求披露实际AI用途与本人贡献。</p></main></html>'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=PROJECT/"outputs")
    parser.add_argument("--bootstrap-repeats",type=int,default=2000)
    parser.add_argument("--bootstrap-seed",type=int,default=20261022)
    parser.add_argument("--batch-size",type=int,default=64)
    args=parser.parse_args();output=args.output.resolve()
    metrics,frames,inputs,alignment,points,validation=load_results(output)
    evidence={}
    for path in (output/"verification.json",output/"threshold_audit.json",output/"duplicate_sensitivity.json"):
        if path.is_file():evidence[path.name]=json.loads(path.read_text(encoding="utf-8"));inputs.append(path)
    summary=aggregate(metrics,["arm","condition","score","policy"])
    within,gaps=metric_changes(metrics);states,transitions=states_and_transitions(frames)
    intervals=paired_bootstrap(frames,args.bootstrap_repeats,args.bootstrap_seed,args.batch_size)
    plots=create_plots(summary,intervals,output)
    document=build_html(summary,within,gaps,intervals,plots,alignment,points,validation,inputs,evidence,args.bootstrap_repeats,args.bootstrap_seed)
    for name,frame in (("summary_mean_sd.csv",summary),("within_arm_changes.csv",within),("cross_arm_gaps.csv",gaps),
                       ("normal_states.csv",states),("policy_transitions.csv",transitions),("paired_intervals.csv",intervals)):
        frame.to_csv(output/name,index=False)
    report=output/"alignment_report_cn.html";report.write_text(document,encoding="utf-8")
    print(json.dumps(dict(status="PASS",test_settings=len(metrics),within_arm_changes=len(within),cross_arm_gaps=len(gaps),
                          normal_states=len(states),policy_transitions=len(transitions),paired_intervals=len(intervals),
                          html_bytes=report.stat().st_size,report=str(report)),ensure_ascii=False))


if __name__=="__main__":main()
