"""Report a fixed-score grid of common IS-validation operating points.

No model inference, fitting or cutoff selection. Run after TEST_COMPLETE.
Only the predeclared 25% comparisons receive paired bootstrap intervals;
10% means remain fully disclosed. No test-selected optimum is produced.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timedelta,timezone
import html,json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

PROJECT=Path(__file__).resolve().parent
PRIOR=PROJECT.parent/"BERT_Rejection_Alignment_20261019"
sys.path.insert(0,str(PROJECT.parent/"BERT_Rejection_Ratio_20261019"))
from report_ratio import ARMS,GRID,CANDIDATES,REFERENCES,ARM_NAMES,METRICS,NAMES,SEEDS,CONDITIONS,CONDITION_NAMES
from report_ratio import booleans,evaluate_predictions,htable,aggregate,sha256,manifest_hashes,counts_to_metrics

TARGETS=(.90,.925,.95,.975,.99)
POLICIES=("aligned_is_acceptance_90","aligned_is_acceptance_92p5","aligned_is_acceptance_95","aligned_is_acceptance_97p5","aligned_is_acceptance_99")
TARGET_BY_POLICY=dict(zip(POLICIES,TARGETS))
BASE_POLICY=POLICIES[2]
BOOTSTRAP_ARMS=("ratio_00","ratio_25","ratio_50")
RATE_DENOMINATORS=(4500,4500,4500,1000,None,1000,4500,5500)
LINEAR=tuple(m for m in METRICS if m!="oos_f1")


def metric_numerator(row,metric):
    return {"in_scope_false_rejection_rate":row["in_scope_false_rejections"],
            "in_scope_correct_and_accepted_rate":row["report_correct_accepted_n"],
            "in_scope_wrong_and_accepted_rate":row["report_wrong_accepted_n"],
            "oos_false_acceptance_rate":1000-row["oos_true_positive"],
            "oos_recall":row["oos_true_positive"],"in_scope_acceptance":4500-row["in_scope_false_rejections"],
            "combined_151_accuracy":row["oos_true_positive"]+row["report_correct_accepted_n"]}[metric]


def load_results(output):
    manifests=[PROJECT/"INPUTS_FROZEN.json",output/"DEVELOPMENT_FROZEN.json",output/"TEST_COMPLETE.json"]
    frozen={}
    for path in manifests:frozen.update(manifest_hashes(json.loads(path.read_text(encoding="utf-8"))))
    metric_path=output/"metrics_by_seed.csv";metrics=pd.read_csv(metric_path,float_precision="round_trip")
    keys=["arm","seed","condition","score","policy"]
    if len(metrics)!=225 or metrics.duplicated(keys).any():raise ValueError("Expected 225 unique test settings")
    for name,values in (("arm",ARMS),("seed",SEEDS),("condition",CONDITIONS),("score",("mahalanobis",)),("policy",POLICIES)):
        if set(metrics[name])!=set(values):raise ValueError(f"Unexpected {name}")
    historical_path=PRIOR/"outputs/metrics_by_seed.csv"
    historical=pd.read_csv(historical_path,float_precision="round_trip")
    historical=historical[historical.policy.eq(BASE_POLICY)].set_index(["arm","seed","condition"])
    inputs=[metric_path,historical_path,*manifests];frames={};canonical=None;shared={}
    identity=["row_id","original_text","true_label","label_id","is_oos"]
    shared_columns=identity+["text","changed","modified_tokens","predicted_label_id"]
    for arm in ARMS:
        frames[arm]={}
        for seed in SEEDS:
            frames[arm][seed]={}
            for condition in CONDITIONS:
                path=output/arm/f"seed{seed}"/f"{condition}_predictions.csv"
                prior_path=PRIOR/"outputs"/arm/f"seed{seed}"/f"{condition}_predictions.csv"
                frame=pd.read_csv(path,float_precision="round_trip");old=pd.read_csv(prior_path,float_precision="round_trip")
                needed=set(shared_columns+["mahalanobis_score"]+[f"mahalanobis_{p}_reject" for p in POLICIES])
                if not needed.issubset(frame) or frame[list(needed)].isna().any().any() or frame.row_id.duplicated().any():raise ValueError("Invalid prediction schema")
                if not set(old).issubset(frame) or not frame[old.columns].equals(old):raise ValueError("Cached alignment columns changed")
                frame["is_oos"],frame["changed"]=booleans(frame.is_oos),booleans(frame.changed)
                if canonical is None:canonical=frame[identity].copy()
                else:
                    if set(frame.row_id)!=set(canonical.row_id):raise ValueError("Original row IDs changed")
                    frame=frame.set_index("row_id").loc[canonical.row_id].reset_index()
                    if not frame[identity].equals(canonical):raise ValueError("Labels or original texts differ")
                if arm==ARMS[0]:shared[(seed,condition)]=frame[shared_columns].copy()
                elif not frame[shared_columns].equals(shared[(seed,condition)]):raise ValueError("Fixed classifier predictions changed across arms")
                labels,oos,pred=frame.label_id.to_numpy(),frame.is_oos.to_numpy(),frame.predicted_label_id.to_numpy()
                unique,counts=np.unique(labels[~oos],return_counts=True)
                if len(frame)!=5500 or oos.sum()!=1000 or not np.array_equal(unique,np.arange(150)) or not np.all(counts==30):raise ValueError("Unexpected evaluation population")
                scores=frame.mahalanobis_score.to_numpy(dtype=float)
                if not np.isfinite(scores).all():raise ValueError("Invalid scores")
                closed=float((pred[~oos]==labels[~oos]).mean());macro=float(f1_score(labels[~oos],pred[~oos],labels=np.arange(150),average="macro"))
                previous_rejected=None;previous_cutoff=None
                for policy in POLICIES:
                    mask=metrics.arm.eq(arm)&metrics.seed.eq(seed)&metrics.condition.eq(condition)&metrics.policy.eq(policy)
                    row=metrics.loc[mask].iloc[0]
                    frozen_group=metrics[metrics.arm.eq(arm)&metrics.seed.eq(seed)&metrics.policy.eq(policy)]
                    if frozen_group.threshold.nunique()!=1:raise ValueError("Cutoff changed across test conditions")
                    rejected=booleans(frame[f"mahalanobis_{policy}_reject"]);frame[f"mahalanobis_{policy}_reject"]=rejected
                    if not np.array_equal(rejected,scores>=float(row.threshold)):raise ValueError("Decision/cutoff mismatch")
                    if previous_rejected is not None and (float(row.threshold)<previous_cutoff or np.any(rejected&~previous_rejected)):
                        raise ValueError("Higher requested acceptance unexpectedly adds rejection")
                    previous_rejected,previous_cutoff=rejected,float(row.threshold)
                    calculated=evaluate_predictions(labels,pred,oos,scores,rejected)
                    calculated.update(closed_set_accuracy=closed,closed_set_macro_f1=macro)
                    for metric,value in calculated.items():
                        if not np.isclose(value,float(row[metric]),rtol=1e-8,atol=1e-10,equal_nan=True):raise ValueError(f"Metric mismatch: {arm}/{seed}/{condition}/{policy}/{metric}")
                    ac=int((~oos&~rejected&(pred==labels)).sum());aw=int((~oos&~rejected&(pred!=labels)).sum())
                    for key,value in (("in_scope_wrong_and_accepted_rate",aw/4500),("oos_false_acceptance_rate",float((oos&~rejected).sum()/1000))):
                        if not np.isclose(float(row[key]),value,rtol=1e-8,atol=1e-10):raise ValueError(f"Wrong acceptance metric mismatch: {key}")
                    metrics.loc[mask,"report_correct_accepted_n"]=ac;metrics.loc[mask,"report_wrong_accepted_n"]=aw
                    metrics.loc[mask,"in_scope_wrong_and_accepted_rate"]=aw/4500
                    metrics.loc[mask,"oos_false_acceptance_rate"]=float((oos&~rejected).sum()/1000)
                    if policy==BASE_POLICY:
                        old_metrics=historical.loc[(arm,seed,condition)]
                        if float(row.threshold)!=float(old_metrics.threshold):raise ValueError("Historical 95% threshold changed")
                        for metric in calculated:
                            if not np.isclose(float(row[metric]),float(old_metrics[metric]),rtol=1e-8,atol=1e-10,equal_nan=True):raise ValueError("Historical 95% result changed")
                frames[arm][seed][condition]=frame;inputs.extend([path,prior_path])
    for _,group in metrics.groupby(["seed","condition"]):
        if group.closed_set_accuracy.nunique()!=1 or group.closed_set_macro_f1.nunique()!=1:raise ValueError("Closed classifier metrics changed")
    for path in inputs:
        expected=frozen.get(str(path.resolve()))
        if expected and sha256(path)!=expected:raise ValueError(f"Frozen file changed: {path}")
    validation_path=output/"validation_metrics_by_seed.csv";validation=pd.read_csv(validation_path,float_precision="round_trip")
    if len(validation)!=75:raise ValueError("Expected 75 validation settings")
    for policy in POLICIES:
        values=validation.loc[validation.policy.eq(policy),"in_scope_acceptance"]
        if values.nunique()!=1 or values.iloc[0]<TARGET_BY_POLICY[policy]:raise ValueError("Validation point not jointly aligned")
    inputs.append(validation_path)
    return metrics,frames,inputs,validation


def gap_tables(metrics):
    indexed=metrics.set_index(["arm","seed","condition","policy"]);rows=[]
    for candidate in CANDIDATES:
        for reference in REFERENCES:
            for seed in SEEDS:
                for condition in CONDITIONS:
                    for policy in POLICIES:
                        c,r=indexed.loc[(candidate,seed,condition,policy)],indexed.loc[(reference,seed,condition,policy)]
                        c95,r95=indexed.loc[(candidate,seed,condition,BASE_POLICY)],indexed.loc[(reference,seed,condition,BASE_POLICY)]
                        for mi,metric in enumerate(METRICS):
                            denominator=RATE_DENOMINATORS[mi]
                            if denominator:
                                count=int(metric_numerator(c,metric))-int(metric_numerator(r,metric))
                                count95=int(metric_numerator(c95,metric))-int(metric_numerator(r95,metric))
                                gap,gap95,change=count/denominator,count95/denominator,(count-count95)/denominator
                            else:count=count95=np.nan;gap=float(c[metric]-r[metric]);gap95=float(c95[metric]-r95[metric]);change=gap-gap95
                            rows.append(dict(candidate=candidate,reference=reference,seed=seed,condition=condition,score="mahalanobis",policy=policy,
                                             target_acceptance=TARGET_BY_POLICY[policy],metric=metric,gap=gap,gap_pp=100*gap,gap_at95=gap95,gap_at95_pp=100*gap95,
                                             gap_change_from95=change,gap_change_from95_pp=100*change,gap_count_difference=count,
                                             gap95_count_difference=count95,rate_denominator=denominator))
    raw=pd.DataFrame(rows);keys=["candidate","reference","condition","score","policy","target_acceptance","metric"]
    summary=aggregate(raw,keys)
    # Means of linear rate contrasts use integer total differences, ensuring exact zero.
    for idx,row in summary.iterrows():
        if row.metric in LINEAR:
            group=raw
            for key in keys:group=group[group[key].eq(row[key])]
            denominator=int(group.rate_denominator.iloc[0])*len(SEEDS)
            count=int(group.gap_count_difference.sum());count95=int(group.gap95_count_difference.sum())
            for metric,value in (("gap",count/denominator),("gap_at95",count95/denominator),("gap_change_from95",(count-count95)/denominator)):
                summary.loc[idx,metric+"_mean"]=value;summary.loc[idx,metric+"_pp_mean"]=100*value
    return raw,summary


def guardrails(summary,metrics):
    table=summary.set_index(["arm","condition","score","policy"]);rows=[]
    for candidate in CANDIDATES:
        for reference in ("ratio_00","ratio_50","clean_reference"):
            for policy in POLICIES:
                local=[]
                for metric in ("oos_f1","oos_recall","in_scope_acceptance"):
                    before,after=(float(table.loc[(arm,"clean","mahalanobis",policy)][metric+"_mean"]) for arm in (reference,candidate))
                    difference=after-before
                    if metric in LINEAR:
                        selected=metrics[metrics.condition.eq("clean")&metrics.policy.eq(policy)]
                        count=lambda arm:sum(int(metric_numerator(row,metric)) for _,row in selected[selected.arm.eq(arm)].iterrows())
                        difference=(count(candidate)-count(reference))/(3*RATE_DENOMINATORS[METRICS.index(metric)])
                    local.append(dict(candidate=candidate,reference=reference,policy=policy,target_acceptance=TARGET_BY_POLICY[policy],condition="clean",
                                      metric=metric,reference_mean=before,candidate_mean=after,delta_pp=100*difference,mean_non_decline=difference>=0))
                passed=all(r["mean_non_decline"] for r in local)
                for row in local:row["all_three_pass"]=passed
                rows.extend(local)
    return pd.DataFrame(rows)


def ordering_checks(metrics):
    """Exact noisy ordering, per seed and across the three fixed seeds."""
    rows=[]
    aliases={"fr":"in_scope_false_rejection_rate","wrong_is_accept":"in_scope_wrong_and_accepted_rate","oos_false_accept":"oos_false_acceptance_rate"}
    for condition in CONDITIONS[1:]:
        for policy in POLICIES:
            for seed in (*SEEDS,None):
                selected=metrics[metrics.condition.eq(condition)&metrics.policy.eq(policy)]
                if seed is not None:selected=selected[selected.seed.eq(seed)]
                nseeds=1 if seed is not None else 3
                row=dict(scope="seed" if seed is not None else "mean",seed=seed,condition=condition,policy=policy,target_acceptance=TARGET_BY_POLICY[policy])
                for alias,metric in aliases.items():
                    count=lambda arm:sum(int(metric_numerator(r,metric)) for _,r in selected[selected.arm.eq(arm)].iterrows())
                    base,candidate,high=(count(arm) for arm in ("ratio_00","ratio_25","ratio_50"))
                    d0,d50=candidate-base,candidate-high;denominator=nseeds*RATE_DENOMINATORS[METRICS.index(metric)]
                    row.update({alias+"_gap25_minus00_pp":100*d0/denominator,alias+"_gap25_minus50_pp":100*d50/denominator,
                                alias+"_between_00_and_50":d0*d50<=0})
                rows.append(row)
    return pd.DataFrame(rows)


def paired_bootstrap(frames,repeats=2000,random_seed=20261023,batch_size=64):
    if repeats<2 or batch_size<1:raise ValueError("Invalid bootstrap configuration")
    first=frames[ARMS[0]][SEEDS[0]]["clean"];labels,oos=first.label_id.to_numpy(),first.is_oos.to_numpy()
    strata=[np.flatnonzero(labels==label) for label in np.unique(labels)]
    events=np.zeros((3,3,3,5,4,5500),dtype=float)
    for ai,arm in enumerate(BOOTSTRAP_ARMS):
        for si,seed in enumerate(SEEDS):
            for ci,condition in enumerate(CONDITIONS):
                frame=frames[arm][seed][condition];correct=(frame.predicted_label_id.to_numpy()==labels)&~oos
                for pi,policy in enumerate(POLICIES):
                    rejected=frame[f"mahalanobis_{policy}_reject"].to_numpy()
                    events[ai,si,ci,pi]=np.stack((rejected&oos,rejected&~oos,~rejected&correct,~rejected&~correct&~oos))
    def numerators(counts):
        tp,fp,ac,aw=(counts[...,i,:] for i in range(4))
        return np.stack((fp,ac,aw,1000-tp,np.zeros_like(tp),tp,4500-fp,tp+ac),axis=-2).sum(axis=1)
    counts=events.sum(axis=-1)[...,None];point=counts_to_metrics(counts).mean(axis=1)[...,0];point_nums=numerators(counts)[...,0]
    draws=np.empty((*point.shape,repeats));draw_nums=np.empty_like(draws);rng=np.random.default_rng(random_seed)
    with threadpool_limits(limits=4):
        for start in range(0,repeats,batch_size):
            size=min(batch_size,repeats-start)
            sample=np.concatenate([rng.choice(indices,size=(size,len(indices)),replace=True) for indices in strata],axis=1)
            weights=np.bincount((sample+5500*np.arange(size)[:,None]).ravel(),minlength=size*5500).reshape(size,5500).astype(float)
            counts=(events.reshape(-1,5500)@weights.T).reshape(*events.shape[:-1],size)
            draws[...,start:start+size]=counts_to_metrics(counts).mean(axis=1);draw_nums[...,start:start+size]=numerators(counts)
    rows=[];target=BOOTSTRAP_ARMS.index("ratio_25")
    for reference in REFERENCES:
        ri=BOOTSTRAP_ARMS.index(reference)
        for ci,condition in enumerate(CONDITIONS):
            for pi,policy in enumerate(POLICIES):
                for mi,metric in enumerate(METRICS):
                    denominator=RATE_DENOMINATORS[mi]
                    if denominator:
                        denominator*=len(SEEDS)
                        n=point_nums[target,ci,pi,mi]-point_nums[ri,ci,pi,mi];n95=point_nums[target,ci,2,mi]-point_nums[ri,ci,2,mi]
                        d=draw_nums[target,ci,pi,mi]-draw_nums[ri,ci,pi,mi];d95=draw_nums[target,ci,2,mi]-draw_nums[ri,ci,2,mi]
                        gap,change=n/denominator,(n-n95)/denominator;gapdraw,changedraw=d/denominator,(d-d95)/denominator
                    else:
                        gap=point[target,ci,pi,mi]-point[ri,ci,pi,mi];gap95=point[target,ci,2,mi]-point[ri,ci,2,mi]
                        gapdraw=draws[target,ci,pi,mi]-draws[ri,ci,pi,mi];gap95draw=draws[target,ci,2,mi]-draws[ri,ci,2,mi]
                        change=gap-gap95;changedraw=gapdraw-gap95draw
                    for estimand,value,dist in (("gap",gap,gapdraw),("gap_change_from95",change,changedraw)):
                        low,high=np.quantile(dist,[.025,.975])
                        rows.append(dict(candidate="ratio_25",reference=reference,condition=condition,score="mahalanobis",policy=policy,
                                         target_acceptance=TARGET_BY_POLICY[policy],metric=metric,estimand=estimand,
                                         estimate=value,ci_low=low,ci_high=high,estimate_pp=100*value,ci_low_pp=100*low,ci_high_pp=100*high,
                                         bootstrap_repeats=repeats,bootstrap_seed=random_seed,n_is=4500,n_oos=1000,n_fixed_models=3,
                                         scope="fixed scores, distributions and all threshold sets; identical stratified text draw across targets/arms/seeds/conditions"))
    return pd.DataFrame(rows)


def stability_table(gap_summary,intervals):
    rows=[]
    for keys,group in gap_summary.groupby(["candidate","reference","condition","metric"],sort=False):
        candidate,reference,condition,metric=keys;values=group.gap_pp_mean.to_numpy()
        sign="all_positive" if np.all(values>0) else "all_negative" if np.all(values<0) else "all_zero" if np.all(values==0) else "includes_zero_or_changes_sign"
        ci=intervals[(intervals.candidate==candidate)&(intervals.reference==reference)&(intervals.condition==condition)&(intervals.metric==metric)&intervals.estimand.eq("gap")]
        interval_direction="not_bootstrapped_10_percent_supplement"
        if len(ci):interval_direction="all_above_zero" if (ci.ci_low_pp>0).all() else "all_below_zero" if (ci.ci_high_pp<0).all() else "not_same_strict_interval_direction_at_every_target"
        rows.append(dict(candidate=candidate,reference=reference,condition=condition,metric=metric,direction_across_targets=sign,
                         minimum_gap_pp=float(values.min()),maximum_gap_pp=float(values.max()),
                         max_absolute_change_from95_pp=float(group.gap_change_from95_pp_mean.abs().max()),interval_direction=interval_direction))
    return pd.DataFrame(rows)


def create_plots(summary,intervals,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    table=summary.set_index(["arm","condition","score","policy"]);x=np.array(TARGETS)*100
    colors={"ratio_00":"#657b8d","ratio_10":"#66a28c","ratio_25":"#c27b35","ratio_50":"#6e669f"}
    panels=(("in_scope_false_rejection_rate","IS false rejection (%)"),("in_scope_wrong_and_accepted_rate","Wrong-head accepted / all IS (%)"),("oos_false_acceptance_rate","OOS false acceptance (%)"))
    fig,axes=plt.subplots(3,3,figsize=(15,11),constrained_layout=True)
    for r,condition in enumerate(CONDITIONS):
        for col,(metric,label) in enumerate(panels):
            ax=axes[r,col]
            for arm in GRID:
                ax.plot(x,[100*table.loc[(arm,condition,"mahalanobis",p)][metric+"_mean"] for p in POLICIES],marker="o",color=colors[arm],label=arm[-2:]+"% noisy fit")
            ax.set_xticks(x);ax.set_xlabel("Requested clean-validation IS acceptance (%)");ax.set_ylabel(label)
            ax.set_title(condition.replace("typo_","").title());ax.spines[["top","right"]].set_visible(False)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc="outside lower center",ncol=4)
    fig.suptitle("Full predeclared operating-point grid | fixed scores | 3-seed means")
    tradeoff=output/"operating_point_tradeoffs.png";fig.savefig(tradeoff,dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4.8),constrained_layout=True)
    for ax,metric in zip(axes,("oos_f1","oos_recall","in_scope_acceptance")):
        for arm in GRID:
            ax.plot(x,[100*table.loc[(arm,"clean","mahalanobis",p)][metric+"_mean"] for p in POLICIES],marker="o",color=colors[arm],label=arm[-2:]+"% noisy fit")
        ax.set_xticks(x);ax.set_xlabel("Requested validation IS acceptance (%)");ax.set_ylabel("Clean test (%)");ax.set_title(metric.replace("_"," "))
        ax.spines[["top","right"]].set_visible(False)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="outside lower center",ncol=4)
    fig.suptitle("Clean metrics at each common validation operating point; no selected optimum")
    clean=output/"operating_point_clean_metrics.png";fig.savefig(clean,dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    for r,reference in enumerate(REFERENCES):
        for col,(metric,label) in enumerate(panels):
            ax=axes[r,col]
            for color,condition in zip(("#397c91","#c27b35","#6e669f"),CONDITIONS):
                part=intervals[(intervals.reference==reference)&(intervals.condition==condition)&(intervals.metric==metric)&intervals.estimand.eq("gap")].sort_values("target_acceptance")
                ax.plot(x,part.estimate_pp,color=color,marker="o",label=condition.replace("typo_","").title())
                ax.fill_between(x,part.ci_low_pp,part.ci_high_pp,color=color,alpha=.13)
            ax.axhline(0,color="#8b999f",ls="--",lw=1);ax.set_xticks(x);ax.set_xlabel("Requested validation IS acceptance (%)")
            ax.set_ylabel("25% minus "+reference[-2:]+"% gap (pp)");ax.set_title(label.replace(" (%)",""));ax.spines[["top","right"]].set_visible(False)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc="outside lower center",ncol=3)
    fig.suptitle("Primary 25% contrasts | pointwise paired 95% CIs (not simultaneous bands)")
    gaps=output/"operating_point_gap_intervals.png";fig.savefig(gaps,dpi=170);plt.close(fig)
    return tradeoff,clean,gaps


def evidence_section(evidence,inputs):
    parts=[]
    if "verification.json" in evidence:
        v=evidence["verification.json"]
        if v.get("status")!="PASS":raise ValueError("Independent verification failed")
        if v.get("metrics_sha256") and v["metrics_sha256"]!=sha256(inputs[0]):raise ValueError("Verification uses different metrics")
        report_status=v.get("report_checks",{}).get("status","未完成")
        parts.append(f'<p>独立主验证PASS：{v["verified_test_settings"]}组test、{v["verified_validation_settings"]}组validation、75个cutoff及95%精确回归；'
                     f'{v["acceptance_monotonicity_adjacent_pairs"]}组相邻接受集合包含关系通过。派生表检查：{html.escape(report_status)}。<a href="verification.json">完整验证记录</a>。</p>')
        checks=v.get("report_checks",{})
        if checks.get("all_480_bootstrap_quantiles_independently_resampled"):
            for name,expected in checks.get("report_artifact_sha256",{}).items():
                if sha256(inputs[0].parent/name)!=expected:raise ValueError(f"Verified report table changed: {name}")
            parts.append(f'<p>独立重抽全部{checks["primary_intervals"]}条CI分位数通过（{checks["bootstrap_repeats"]}次、seed {checks["bootstrap_seed"]}）；'
                         f'95%基准的{checks["identically_zero_95_change_intervals"]}条gap变化区间精确为[0,0]。'
                         f'{checks["strict_clean_guardrails"]}条整数率保留判断与{checks["ordering_checks"]["rows"]}条排序记录也通过独立核验。</p>')
    if "duplicate_sensitivity.json" in evidence:
        s=evidence["duplicate_sensitivity.json"]
        parts.append(f'<p>排除既有train–test重叠row599/938后，450行评估复算；主gap最大变化{float(s["max_absolute_exclusion_change_in_gap_pp"]):.6f} pp，'
                     f'gap相对95%的变化最大改变{float(s["max_absolute_exclusion_change_in_gap_minus_95_gap_pp"]):.6f} pp。该敏感性不撤销历史测试曝光。</p>')
    for name,label in (("threshold_audit.json","独立IS-only阈值审计"),("duplicate_sensitivity.json","重复文本敏感性")):
        if name in evidence:parts.append(f'<p><a href="{name}">{label}</a>：使用冻结结果，未据检查结果重新选择运行点。</p>')
    return "".join(parts)


def build_html(summary,gap_summary,guardrail,stability,ordering,intervals,validation,plots,inputs,evidence,repeats,random_seed):
    table=summary.set_index(["arm","condition","score","policy"])
    ci=intervals.set_index(["reference","condition","policy","metric","estimand"])
    mean_ordering=ordering[ordering.scope.eq("mean")];seed_ordering=ordering[ordering.scope.eq("seed")]
    aliases=("fr","wrong_is_accept","oos_false_accept")
    all_mean_between=all(mean_ordering[a+"_between_00_and_50"].all() for a in aliases)
    def signs(alias,reference):
        values=seed_ordering[f"{alias}_gap25_minus{reference}_pp"]
        return f"{int((values>0).sum())}正 / {int((values==0).sum())}零 / {int((values<0).sum())}负"
    order_rows=[[label,signs(alias,"00"),signs(alias,"50")] for alias,label in zip(aliases,("IS误拒率","已知类错分类放行率","OOS漏放率"))]
    exception_rows=[]
    for row in seed_ordering.itertuples():
        for reference,expected_sign in (("00",1),("50",-1)):
            value=getattr(row,f"oos_false_accept_gap25_minus{reference}_pp")
            if value*expected_sign<=0:
                exception_rows.append([int(row.seed),CONDITION_NAMES[row.condition],f"{100*row.target_acceptance:g}%",f"25%−{reference}%",f"{value:+.3f}"])
    def mean_value(condition,policy,metric):return 100*float(table.loc[("ratio_25",condition,"mahalanobis",policy)][metric+"_mean"])
    endpoint_rows=[[CONDITION_NAMES[c],f"{mean_value(c,POLICIES[0],METRICS[0]):.3f} → {mean_value(c,POLICIES[-1],METRICS[0]):.3f}",
                    f"{mean_value(c,POLICIES[0],METRICS[2]):.3f} → {mean_value(c,POLICIES[-1],METRICS[2]):.3f}",
                    f"{mean_value(c,POLICIES[0],METRICS[3]):.3f} → {mean_value(c,POLICIES[-1],METRICS[3]):.3f}"] for c in CONDITIONS]
    magnitude_rows=[]
    for metric in (METRICS[2],METRICS[3]):
        for policy in (POLICIES[0],POLICIES[2],POLICIES[4]):
            row=ci.loc[("ratio_00","typo_medium",policy,metric,"gap")]
            change=ci.loc[("ratio_00","typo_medium",policy,metric,"gap_change_from95")]
            magnitude_rows.append([NAMES[metric],f"{100*TARGET_BY_POLICY[policy]:g}%",f"{row.estimate_pp:+.3f} [{row.ci_low_pp:+.3f}, {row.ci_high_pp:+.3f}]",
                                   f"{change.estimate_pp:+.3f} [{change.ci_low_pp:+.3f}, {change.ci_high_pp:+.3f}]"])
    points=[]
    for policy in POLICIES:
        rows=validation[validation.policy.eq(policy)];actual=float(rows.in_scope_acceptance.iloc[0])
        points.append([f"{100*TARGET_BY_POLICY[policy]:g}%",int(round(TARGET_BY_POLICY[policy]*3000)),int(round(actual*3000)),f"{100*actual:.4f}%"])
    full_rows=[];primary_rows=[];guard_rows=[];stability_rows=[]
    for policy in POLICIES:
        for arm in ARMS:
            for condition in CONDITIONS:
                row=table.loc[(arm,condition,"mahalanobis",policy)]
                full_rows.append([f"{100*TARGET_BY_POLICY[policy]:g}%",ARM_NAMES[arm],CONDITION_NAMES[condition],*[f"{100*row[m+'_mean']:.3f}" for m in METRICS]])
    for reference in REFERENCES:
        for condition in CONDITIONS:
            for policy in POLICIES:
                for metric in METRICS[:4]:
                    gap,change=(ci.loc[(reference,condition,policy,metric,e)] for e in ("gap","gap_change_from95"))
                    primary_rows.append([ARM_NAMES[reference],CONDITION_NAMES[condition],f"{100*TARGET_BY_POLICY[policy]:g}%",NAMES[metric],
                                         f"{gap.estimate_pp:+.4f} [{gap.ci_low_pp:+.4f}, {gap.ci_high_pp:+.4f}]",
                                         f"{change.estimate_pp:+.4f} [{change.ci_low_pp:+.4f}, {change.ci_high_pp:+.4f}]"])
    for _,group in guardrail.groupby(["candidate","reference","policy"],sort=False):
        row=group.iloc[0];values=group.set_index("metric")
        guard_rows.append([ARM_NAMES[row.candidate],ARM_NAMES[row.reference],f"{100*row.target_acceptance:g}%",
                           *[f"{values.loc[m,'delta_pp']:+.4f}" for m in ("oos_f1","oos_recall","in_scope_acceptance")],
                           "三项均未下降" if row.all_three_pass else "未全部保留"])
    direction_names={"all_positive":"五点均为正","all_negative":"五点均为负","all_zero":"五点均为0","includes_zero_or_changes_sign":"含0或变号"}
    interval_names={"all_above_zero":"五点区间均高于0","all_below_zero":"五点区间均低于0","not_same_strict_interval_direction_at_every_target":"不是五点均严格同向","not_bootstrapped_10_percent_supplement":"10%仅作均值补充"}
    for row in stability.itertuples():
        if row.candidate=="ratio_25" and row.metric in METRICS[:4]:
            stability_rows.append([ARM_NAMES[row.reference],CONDITION_NAMES[row.condition],NAMES[row.metric],direction_names[row.direction_across_targets],
                                   f"[{row.minimum_gap_pp:+.3f}, {row.maximum_gap_pp:+.3f}]",f"{row.max_absolute_change_from95_pp:.3f}",interval_names[row.interval_direction]])
    image=lambda p:f'<img src="{html.escape(p.name,quote=True)}" alt="{html.escape(p.stem)}">'
    now=datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M UTC+08:00")
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>共同验证接受率运行点实验</title>
<style>*{{box-sizing:border-box}}body{{margin:0;background:#f3f6f7;color:#193746;font:16px/1.75 system-ui,"Microsoft YaHei",sans-serif}}main{{max-width:1300px;margin:26px auto;padding:34px 40px;background:#fff;border:1px solid #dbe5e8;border-radius:16px}}
h1{{font-size:29px;line-height:1.4}}h2{{font-size:23px;margin-top:32px}}.meta{{font-size:13px;color:#5c7280}}.lead{{background:#fff5e9;border-left:5px solid #c27b35;padding:16px 20px}}
.table-scroll{{overflow-x:auto;margin:16px 0}}table{{border-collapse:collapse;width:100%;font-size:12px;font-variant-numeric:tabular-nums}}th,td{{padding:8px 9px;border-bottom:1px solid #dce5e8;text-align:left;vertical-align:top}}th{{background:#eaf0f3;white-space:nowrap}}td{{overflow-wrap:anywhere}}
img{{max-width:100%;height:auto;margin:16px 0;border:1px solid #dce5e8;border-radius:8px}}summary{{cursor:pointer;font-weight:650}}@media(max-width:700px){{main{{padding:24px 16px;margin:0;border-radius:0}}}}</style>
<main><p class="meta">MFE310TC · 固定模型/分数的共同验证运行点 · {now}</p><h1>90%到99%的共同接受率：比例差异是否稳定？</h1>
<p class="lead">{('在两个noisy条件的全部五点，25%拟合噪声的三项均值都位于0%与50%之间：比0%少误拒，但多放过已知类错分类和OOS；比50%则相反。' if all_mean_between else '逐点排序并非全部成立，需按下表分条件判断。')}
这种折中方向没有在五点内消失，但效应幅度随运行点改变，clean差值也会换向；不能称为全面保留或最优比例。99%接受目标下，误拒下降伴随明显的绝对错误放行代价。</p>
<p>固定BERT、拟合分布与逐样本分数，完整比较90%、92.5%、95%、97.5%、99%五个预先指定运行点；主比较25%对0%/50%，10%保留全部均值，不据test选择目标或比例。</p>
<h2>共同接受率如何实现？</h2><p>只用每个detector的3,000条clean validation IS分数选阈值，75个新旧运行点全部冻结后重新阈值化已有test分数。
所有15模型在每个target共同使用可实现的整数接受数；若ties妨碍目标值，则按预先规则上移到最近共同可行数。
95%列复用并逐条核对上一轮。全部旧列、分类头和距离分数不变，未做模型推理、训练或重新拟合。</p>
{htable(['目标接受率','目标IS接受数/3000','实际共同接受数/3000','实际共同接受率'],points)}
<p>目标是验证集约束，不保证clean或noisy test达到相同接受率。高目标在同一detector上提高阈值、减少拒绝，也可能放过更多OOS和已知类错分类。
原始阈值不能跨不同协方差直接比较宽松程度；历史15k clean单列参考，比例比较使用固定30k视图。</p>
<p>99%运行点仅留下30条validation IS拒绝，属于更靠近分布尾端的分位数；本实验区间没有重抽validation来计入这个阈值的不确定性，也不外推五个已测点之外。</p>
<h2>完整运行点曲线：误拒与两种错放</h2>{image(plots[0])}
<p>IS误拒、正确且接受、已知类错误分类却接受均以4,500条IS为分母；OOS错误接受以1,000条OOS为分母，等于1−OOS recall。
错误分类被接受不算恢复正确服务。曲线为三个固定seed均值；不是运行点最优性曲线，不用test选择部署策略。</p>
{htable(['25%拟合臂','误拒率：90→99 %','已知类错分类放行：90→99 %','OOS漏放：90→99 %'],endpoint_rows)}
<h2>均值排序与逐seed排序要分开</h2><p>下表统计两个noisy条件×五target×三个seed，共30个有符号差值；三seed均值的10点排序不能自动推广到每个seed。</p>
{htable(['指标','25%−0%：30个seed差值','25%−50%：30个seed差值'],order_rows)}
<details><summary>OOS逐seed零差或反向记录</summary>{htable(['seed','条件','target','比较','OOS漏放差 pp'],exception_rows)}</details>
<p>FR与已知类错分类放行的方向在全部seed保持；OOS存在零差和反向个例。排序说的是本组固定模型和已测五点，不能作为每次训练或所有阈值的保证。</p>
<h2>主25%比较：稳定必须分三个层面</h2>{image(plots[2])}
<p>“方向稳定”只看五点均值是否同号；“幅度稳定”看gap范围及相对95%的最大变化；“区间稳定”看各点配对区间是否始终排除0且同向。
CI跨0或接触0不证明相等，方向同号也不说明幅度不变。图中阴影是逐点(pointwise)95%条件CI，连接区间端点不构成同时置信带。
点间关联由相同重抽样保留，不能把五个点当独立重复实验。</p>
<p>例如中度扰动下25%相对0%的错误放行差值会随target明显变化：</p>
{htable(['中度扰动指标','target','25%−0% gap [95% CI] pp','gap−gap95 [95% CI] pp'],magnitude_rows)}
{htable(['25%减参照','输入条件','指标','均值方向','五点gap范围 pp','相对95最大变化 pp','条件区间方向'],stability_rows)}
<details><summary>每个运行点gap与相对95%的变化：主25%全部IS/OOS代价指标</summary>
{htable(['参照','条件','target','指标','gap [95% CI] pp','gap−gap95 [95% CI] pp'],primary_rows)}</details>
<p><code>gap(target)=M(25%,target)−M(reference,target)</code>；<code>gap_change_from95=gap(target)−gap(95%)</code>。
正负只表示有符号差值的变化，并非自动表示绝对差距缩小；不计算普遍因果归因百分比。95%变化及其区间严格为0。</p>
<h2>clean保留：在相同运行点逐项比较</h2>{image(plots[1])}
<p>以下三项均值相对同target的参照全部不下降，才满足严格描述性保留；没有事后margin，也不是非劣效检验。
相对50%的变化、相对0%的保留和历史clean参考分开列出，不把提高目标接受率带来的变化说成模型改善。</p>
<p>clean误拒及F1差值在不同运行点会换向；没有一个跨target的“全面保clean”结论。线性率的保留判定使用三seed整数计数差，数学相等时严格记为0。</p>
<details><summary>所有候选/target的clean三项差值与严格保留判断</summary>{htable(['候选','同target参照','target','OOS F1差 pp','OOS recall差 pp','IS接受率差 pp','三项mean判断'],guard_rows)}</details>
<h2>配对区间与解释边界</h2><p>{repeats:,}次bootstrap、随机种子{random_seed}；150个IS intent每类有放回抽30条，OOS抽1,000条，
同一原文抽样权重共享于全部target、模型seed、条件和主比较。每seed先计算非线性F1再取三模型均值；线性率先作整数计数差再除统一分母，避免浮点误差把真实0端点变成微小负数。</p>
<p>仅25%主比较计算CI，10%为完整均值补充。gap相对95%的区间直接由同次抽样中的差得到，不能相减两套区间端点。
所有区间条件于固定模型、分数、分布、来源子集和阈值；不包括验证阈值选择、重新训练/拟合、新噪声或新子集的不确定性，多比较未校正。</p>
<p>95%回归指阈值、逐样本决定和点估计精确一致。本轮bootstrap随机种子与上一轮不同，有限2,000次抽样会使CI端点略有变化；
尤其接近0的端点可能触0或略离0，不应把这种边界变化当成新的模型证据。</p>
<p>此前test发现推动本轮五点检查，仍属重复测试后的探索工作。合成扰动继承原标签，未逐条人工确认语义保持；不推出自然拼写泛化。
原train–validation/test重叠仍披露，排除599/938也不能撤销历史曝光。观察到的运行点敏感性不是普遍因果百分比。</p>
<h2>核验与文件</h2>{evidence_section(evidence,inputs)}
<p><a href="primary_paired_intervals.csv">全部480条主比较CI（含F1/recall）</a> · <a href="gap_summary.csv">10%与25%全部gap均值/SD</a> · <a href="stability_summary.csv">稳定性分层记录</a> · <a href="ordering_checks.csv">40行均值/逐seed排序</a> · <a href="clean_guardrails.csv">clean保留表</a> · <a href="../PROTOCOL.json">冻结协议</a></p>
<details><summary>全部五臂/五点/三条件均值（含历史clean参考）</summary>{htable(['target','拟合臂','条件',*[NAMES[m]+' %' for m in METRICS]],full_rows)}</details>
<details><summary>冻结数据及证据SHA256（可更新verification只链接）</summary>{htable(['文件','SHA256'],[[str(p),sha256(p)] for p in inputs if p.name!='verification.json'])}</details>
<p class="meta">图像使用同目录PNG，分享时保留outputs文件夹。AI协助实现与统计核验；学生应核验解释并按课程要求披露实际AI用途与本人贡献。</p></main></html>'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=PROJECT/"outputs")
    parser.add_argument("--bootstrap-repeats",type=int,default=2000)
    parser.add_argument("--bootstrap-seed",type=int,default=20261023)
    parser.add_argument("--batch-size",type=int,default=64)
    parser.add_argument("--html-only",action="store_true",help="Refresh only HTML after final independent verification; preserve every derived CSV and PNG")
    args=parser.parse_args();output=args.output.resolve()
    metrics,frames,inputs,validation=load_results(output);evidence={}
    for path in (output/"verification.json",output/"threshold_audit.json",output/"duplicate_sensitivity.json"):
        if path.is_file():evidence[path.name]=json.loads(path.read_text(encoding="utf-8"));inputs.append(path)
    if args.html_only:
        read=lambda name:pd.read_csv(output/name,float_precision="round_trip")
        summary,gaps,guardrail,stability,ordering,intervals=[read(name) for name in ("summary_mean_sd.csv","gap_summary.csv","clean_guardrails.csv","stability_summary.csv","ordering_checks.csv","primary_paired_intervals.csv")]
        plots=[output/name for name in ("operating_point_tradeoffs.png","operating_point_clean_metrics.png","operating_point_gap_intervals.png")]
        protected=[*output.glob("*.csv"),*output.glob("*.png")];before={str(p):sha256(p) for p in protected}
        document=build_html(summary,gaps,guardrail,stability,ordering,intervals,validation,plots,inputs,evidence,args.bootstrap_repeats,args.bootstrap_seed)
        report=output/"operating_points_report_cn.html";report.write_text(document,encoding="utf-8")
        if before!={str(p):sha256(p) for p in protected}:raise ValueError("HTML-only refresh changed protected output")
        print(json.dumps(dict(status="PASS",mode="html_only",protected_outputs_unchanged=len(protected),html_bytes=report.stat().st_size,report=str(report)),ensure_ascii=False));return
    summary=aggregate(metrics,["arm","condition","score","policy"])
    raw,gaps=gap_tables(metrics);guardrail=guardrails(summary,metrics);ordering=ordering_checks(metrics)
    intervals=paired_bootstrap(frames,args.bootstrap_repeats,args.bootstrap_seed,args.batch_size)
    stability=stability_table(gaps,intervals);plots=create_plots(summary,intervals,output)
    document=build_html(summary,gaps,guardrail,stability,ordering,intervals,validation,plots,inputs,evidence,args.bootstrap_repeats,args.bootstrap_seed)
    for name,frame in (("summary_mean_sd.csv",summary),("gap_by_seed.csv",raw),("gap_summary.csv",gaps),("clean_guardrails.csv",guardrail),
                       ("stability_summary.csv",stability),("primary_paired_intervals.csv",intervals),("ordering_checks.csv",ordering)):
        frame.to_csv(output/name,index=False)
    report=output/"operating_points_report_cn.html";report.write_text(document,encoding="utf-8")
    print(json.dumps(dict(status="PASS",test_settings=len(metrics),gap_by_seed=len(raw),gap_summary=len(gaps),clean_guardrails=len(guardrail),
                          stability_summary=len(stability),primary_intervals=len(intervals),html_bytes=report.stat().st_size,report=str(report)),ensure_ascii=False))


if __name__=="__main__":main()
