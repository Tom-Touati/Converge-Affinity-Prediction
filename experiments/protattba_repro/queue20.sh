#!/usr/bin/env bash
# Queue runner: launches jobs onto two EC2 boxes, respecting a per-box concurrency cap,
# polling until a slot frees rather than piling onto an already-loaded box.
set -uo pipefail
KEY="C:/Users/tomto/.ssh/tom_aws.pem"
SSH="ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=15"
MAX_CONCURRENT=2
COM="--arch sitetok --folds 0 1 2 3 4 --seeds 0 1 2 --clip 4 --select-on per_complex --wd 0.1 --patience 15"
CL="--chem-scale cluster --clusters /home/ubuntu/perturb/row_clusters.csv"
SEQBASE='"pca_dim":128,"proj":64,"hidden":128,"layers":1,"dropout":0.35,"input_noise":0.0,"feature_dropout":0.0,"chem_dim":0,"split_proj":true,"n_heads":2,"mut_pair_ffn":true,"mut_pair_op":"sub","mut_pair_act":true'
GATED='"pca_dim":128,"proj":64,"subtract":true,"use_site_pool":false,"hidden":128,"layers":1,"input_noise":0.25,"feature_dropout":0.35,"dropout":0.35,"chem_dim":39,"mpnn_proj":64,"gated_fusion":true,"split_proj":true'
DXATTN='"pca_dim":128,"proj":64,"subtract":true,"use_site_pool":false,"hidden":128,"layers":1,"input_noise":0.25,"feature_dropout":0.35,"dropout":0.35,"chem_dim":39,"mpnn_proj":64,"delta_xattn":true,"n_heads":2,"split_proj":true,"attn_direction":"struct_to_seq","rope":true,"rope_base":200.0'

# --- queue for 44.211.239.22 (10 items) ---
Q22_NAME=(gcs_seed5a gcs_cluster_split mps_dropout50 mps_dropout15 mps_hidden256 mps_layers2 mps_wd30 mps_wd02 mps_gradclip20 gated_ord)
Q22_SEEDS=("0 1 2 3 4" "0 1 2" "0 1 2" "0 1 2" "0 1 2" "0 1 2" "0 1 2" "0 1 2" "0 1 2" "0 1 2")
Q22_OVR=(
  "{$GATED}"
  "{$GATED}"
  "{$SEQBASE,\"dropout\":0.5}"
  "{$SEQBASE,\"dropout\":0.15}"
  "{$SEQBASE,\"hidden\":256}"
  "{$SEQBASE,\"layers\":2}"
  "{$SEQBASE}"
  "{$SEQBASE}"
  "{$SEQBASE}"
  "{$GATED,\"ordinal\":3}"
)
Q22_WD=(0.1 0.1 0.1 0.1 0.1 0.1 0.3 0.02 0.1 0.1)
Q22_CLIP=(10 10 10 10 10 10 10 10 20 10)
Q22_CL=(1 1 0 0 0 0 0 0 0 1)
Q22_CHEMTABLE=(chem_perturb_v2 chem_perturb_v2 "" "" "" "" "" "" "" chem_perturb_v2)

# --- queue for 44.222.82.18 (10 items) ---
Q18_NAME=(gcf_seed5 mps_cluster_split gcs_lowlr mps_fullchem mps_noise25 mps_featdrop35 mps_ord dxattn_kitchen gcg39_nocluster gcf_gradclip20)
Q18_SEEDS=("0 1 2 3 4" "0 1 2" "0 1 2" "0 1 2" "0 1 2" "0 1 2" "0 1 2" "0 1 2" "0 1 2" "0 1 2")
Q18_OVR=(
  "{$GATED}"
  "{$SEQBASE}"
  "{$GATED}"
  "{$SEQBASE,\"chem_dim\":39}"
  "{$SEQBASE,\"input_noise\":0.25}"
  "{$SEQBASE,\"feature_dropout\":0.35}"
  "{$SEQBASE,\"ordinal\":3}"
  "{$DXATTN}"
  "{$GATED}"
  "{$GATED}"
)
Q18_WD=(0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1)
Q18_CLIP=(10 10 10 10 10 10 10 10 10 20)
Q18_LR=("" "" "--lr 0.0005" "" "" "" "" "" "" "")
Q18_CL=(1 0 1 0 0 0 0 1 0 1)
Q18_CHEMTABLE=(chem_perturb_v2 "" chem_perturb_v2 chem_perturb_v2 "" "" "" chem_perturb_v2 chem_perturb_v2 chem_perturb_v2)

running_count() {
  $SSH ubuntu@"$1" "ps -eo args | grep -c '[_]perturb_v2_colab.py'" 2>/dev/null
}

launch() {
  ip=$1 name=$2 seeds=$3 ovr=$4 wd=$5 clip=$6 cl_flag=$7 lr=$8 ct=$9
  env=""; [ -n "$ct" ] && env="CHEM_TABLE=$ct"
  $SSH ubuntu@"$ip" "
    cd /home/ubuntu/perturb
    PERTURB_ROOT=/home/ubuntu/perturb $env setsid nohup /home/ubuntu/venv/bin/python \
      _perturb_v2_colab.py --exp $name $COM --seeds $seeds $cl_flag $lr \
      --wd $wd --grad-clip $clip --overrides '$ovr' > $name.log 2>&1 < /dev/null &
    " 2>/dev/null
  echo "[$(date -u +%H:%M:%S)] $ip launched $name"
}

if [ "${1:-}" = "--dry-run" ]; then
  for i in "${!Q22_NAME[@]}"; do
    cl=""; [ "${Q22_CL[$i]}" = "1" ] && cl="$CL"
    echo "22[$i] ${Q22_NAME[$i]}  seeds='${Q22_SEEDS[$i]}'  wd=${Q22_WD[$i]}  clip=${Q22_CLIP[$i]}  cl='${cl}'  chemtable='${Q22_CHEMTABLE[$i]}'"
    echo "     ovr: ${Q22_OVR[$i]}"
  done
  for i in "${!Q18_NAME[@]}"; do
    cl=""; [ "${Q18_CL[$i]}" = "1" ] && cl="$CL"
    echo "18[$i] ${Q18_NAME[$i]}  seeds='${Q18_SEEDS[$i]}'  wd=${Q18_WD[$i]}  clip=${Q18_CLIP[$i]}  cl='${cl}'  lr='${Q18_LR[$i]}'  chemtable='${Q18_CHEMTABLE[$i]}'"
    echo "     ovr: ${Q18_OVR[$i]}"
  done
  exit 0
fi

i22=0; i18=0
while [ $i22 -lt 10 ] || [ $i18 -lt 10 ]; do
  if [ $i22 -lt 10 ]; then
    n=$(running_count 44.211.239.22)
    if [ "${n:-9}" -lt $MAX_CONCURRENT ]; then
      cl=""; [ "${Q22_CL[$i22]}" = "1" ] && cl="$CL"
      launch 44.211.239.22 "${Q22_NAME[$i22]}" "${Q22_SEEDS[$i22]}" "${Q22_OVR[$i22]}" \
        "${Q22_WD[$i22]}" "${Q22_CLIP[$i22]}" "$cl" "" "${Q22_CHEMTABLE[$i22]}"
      i22=$((i22+1))
    fi
  fi
  if [ $i18 -lt 10 ]; then
    n=$(running_count 44.222.82.18)
    if [ "${n:-9}" -lt $MAX_CONCURRENT ]; then
      cl=""; [ "${Q18_CL[$i18]}" = "1" ] && cl="$CL"
      launch 44.222.82.18 "${Q18_NAME[$i18]}" "${Q18_SEEDS[$i18]}" "${Q18_OVR[$i18]}" \
        "${Q18_WD[$i18]}" "${Q18_CLIP[$i18]}" "$cl" "${Q18_LR[$i18]}" "${Q18_CHEMTABLE[$i18]}"
      i18=$((i18+1))
    fi
  fi
  sleep 90
done
echo "[$(date -u +%H:%M:%S)] ALL 20 QUEUED"
