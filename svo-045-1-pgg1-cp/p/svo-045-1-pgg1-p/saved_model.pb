Ыс	
мГ
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
Г
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
О
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"serve*2.2.02v2.2.0-0-g2b96f3662b8за
Ц
5ActorDistributionNetwork/EncodingNetwork/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*F
shared_name75ActorDistributionNetwork/EncodingNetwork/dense/kernel
П
IActorDistributionNetwork/EncodingNetwork/dense/kernel/Read/ReadVariableOpReadVariableOp5ActorDistributionNetwork/EncodingNetwork/dense/kernel*
_output_shapes

:d*
dtype0
О
3ActorDistributionNetwork/EncodingNetwork/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*D
shared_name53ActorDistributionNetwork/EncodingNetwork/dense/bias
З
GActorDistributionNetwork/EncodingNetwork/dense/bias/Read/ReadVariableOpReadVariableOp3ActorDistributionNetwork/EncodingNetwork/dense/bias*
_output_shapes
:d*
dtype0
Ъ
7ActorDistributionNetwork/EncodingNetwork/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*H
shared_name97ActorDistributionNetwork/EncodingNetwork/dense_1/kernel
У
KActorDistributionNetwork/EncodingNetwork/dense_1/kernel/Read/ReadVariableOpReadVariableOp7ActorDistributionNetwork/EncodingNetwork/dense_1/kernel*
_output_shapes

:dd*
dtype0
Т
5ActorDistributionNetwork/EncodingNetwork/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*F
shared_name75ActorDistributionNetwork/EncodingNetwork/dense_1/bias
Л
IActorDistributionNetwork/EncodingNetwork/dense_1/bias/Read/ReadVariableOpReadVariableOp5ActorDistributionNetwork/EncodingNetwork/dense_1/bias*
_output_shapes
:d*
dtype0
Ъ
7ActorDistributionNetwork/EncodingNetwork/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*H
shared_name97ActorDistributionNetwork/EncodingNetwork/dense_2/kernel
У
KActorDistributionNetwork/EncodingNetwork/dense_2/kernel/Read/ReadVariableOpReadVariableOp7ActorDistributionNetwork/EncodingNetwork/dense_2/kernel*
_output_shapes

:dd*
dtype0
Т
5ActorDistributionNetwork/EncodingNetwork/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*F
shared_name75ActorDistributionNetwork/EncodingNetwork/dense_2/bias
Л
IActorDistributionNetwork/EncodingNetwork/dense_2/bias/Read/ReadVariableOpReadVariableOp5ActorDistributionNetwork/EncodingNetwork/dense_2/bias*
_output_shapes
:d*
dtype0
є
LActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*]
shared_nameNLActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/kernel
э
`ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/kernel/Read/ReadVariableOpReadVariableOpLActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/kernel*
_output_shapes

:d*
dtype0
ь
JActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*[
shared_nameLJActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/bias
х
^ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/bias/Read/ReadVariableOpReadVariableOpJActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/bias*
_output_shapes
:*
dtype0
P
ConstConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ

NoOpNoOp
О
Const_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*ї
valueэBъ Bу
9
_actor_network
model_variables

signatures
z
_encoder
_projection_networks
trainable_variables
regularization_losses
	variables
		keras_api
8

0
1
2
3
4
5
6
7
 
n
_postprocessing_layers
trainable_variables
regularization_losses
	variables
	keras_api
i
_projection_layer
trainable_variables
regularization_losses
	variables
	keras_api
8

0
1
2
3
4
5
6
7
 
8

0
1
2
3
4
5
6
7
­

layers
non_trainable_variables
trainable_variables
regularization_losses
layer_metrics
	variables
layer_regularization_losses
 metrics
wu
VARIABLE_VALUE5ActorDistributionNetwork/EncodingNetwork/dense/kernel,model_variables/0/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE3ActorDistributionNetwork/EncodingNetwork/dense/bias,model_variables/1/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE7ActorDistributionNetwork/EncodingNetwork/dense_1/kernel,model_variables/2/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE5ActorDistributionNetwork/EncodingNetwork/dense_1/bias,model_variables/3/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE7ActorDistributionNetwork/EncodingNetwork/dense_2/kernel,model_variables/4/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE5ActorDistributionNetwork/EncodingNetwork/dense_2/bias,model_variables/5/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUELActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/kernel,model_variables/6/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEJActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/bias,model_variables/7/.ATTRIBUTES/VARIABLE_VALUE

!0
"1
#2
$3
*

0
1
2
3
4
5
 
*

0
1
2
3
4
5
­

%layers
&non_trainable_variables
trainable_variables
regularization_losses
'layer_metrics
	variables
(layer_regularization_losses
)metrics
h

kernel
bias
*trainable_variables
+regularization_losses
,	variables
-	keras_api

0
1
 

0
1
­

.layers
/non_trainable_variables
trainable_variables
regularization_losses
0layer_metrics
	variables
1layer_regularization_losses
2metrics

0
1
 
 
 
 
R
3trainable_variables
4regularization_losses
5	variables
6	keras_api
h


kernel
bias
7trainable_variables
8regularization_losses
9	variables
:	keras_api
h

kernel
bias
;trainable_variables
<regularization_losses
=	variables
>	keras_api
h

kernel
bias
?trainable_variables
@regularization_losses
A	variables
B	keras_api

!0
"1
#2
$3
 
 
 
 

0
1
 

0
1
­

Clayers
Dnon_trainable_variables
*trainable_variables
+regularization_losses
Elayer_metrics
,	variables
Flayer_regularization_losses
Gmetrics

0
 
 
 
 
 
 
 
­

Hlayers
Inon_trainable_variables
3trainable_variables
4regularization_losses
Jlayer_metrics
5	variables
Klayer_regularization_losses
Lmetrics


0
1
 


0
1
­

Mlayers
Nnon_trainable_variables
7trainable_variables
8regularization_losses
Olayer_metrics
9	variables
Player_regularization_losses
Qmetrics

0
1
 

0
1
­

Rlayers
Snon_trainable_variables
;trainable_variables
<regularization_losses
Tlayer_metrics
=	variables
Ulayer_regularization_losses
Vmetrics

0
1
 

0
1
­

Wlayers
Xnon_trainable_variables
?trainable_variables
@regularization_losses
Ylayer_metrics
A	variables
Zlayer_regularization_losses
[metrics
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
l
action_0/discountPlaceholder*#
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ

action_0/observationPlaceholder*+
_output_shapes
:џџџџџџџџџ*
dtype0* 
shape:џџџџџџџџџ
j
action_0/rewardPlaceholder*#
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
m
action_0/step_typePlaceholder*#
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
М
StatefulPartitionedCallStatefulPartitionedCallaction_0/discountaction_0/observationaction_0/rewardaction_0/step_type5ActorDistributionNetwork/EncodingNetwork/dense/kernel3ActorDistributionNetwork/EncodingNetwork/dense/bias7ActorDistributionNetwork/EncodingNetwork/dense_1/kernel5ActorDistributionNetwork/EncodingNetwork/dense_1/bias7ActorDistributionNetwork/EncodingNetwork/dense_2/kernel5ActorDistributionNetwork/EncodingNetwork/dense_2/biasLActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/kernelJActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/bias*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ**
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*+
f&R$
"__inference_signature_wrapper_1806
]
get_initial_state_batch_sizePlaceholder*
_output_shapes
: *
dtype0*
shape: 
е
PartitionedCallPartitionedCallget_initial_state_batch_size*
Tin
2*

Tout
 * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*+
f&R$
"__inference_signature_wrapper_1818
л
PartitionedCall_1PartitionedCallConst*
Tin
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*+
f&R$
"__inference_signature_wrapper_1833
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameIActorDistributionNetwork/EncodingNetwork/dense/kernel/Read/ReadVariableOpGActorDistributionNetwork/EncodingNetwork/dense/bias/Read/ReadVariableOpKActorDistributionNetwork/EncodingNetwork/dense_1/kernel/Read/ReadVariableOpIActorDistributionNetwork/EncodingNetwork/dense_1/bias/Read/ReadVariableOpKActorDistributionNetwork/EncodingNetwork/dense_2/kernel/Read/ReadVariableOpIActorDistributionNetwork/EncodingNetwork/dense_2/bias/Read/ReadVariableOp`ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/kernel/Read/ReadVariableOp^ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/bias/Read/ReadVariableOpConst_1*
Tin
2
*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*&
f!R
__inference__traced_save_1915
р
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename5ActorDistributionNetwork/EncodingNetwork/dense/kernel3ActorDistributionNetwork/EncodingNetwork/dense/bias7ActorDistributionNetwork/EncodingNetwork/dense_1/kernel5ActorDistributionNetwork/EncodingNetwork/dense_1/bias7ActorDistributionNetwork/EncodingNetwork/dense_2/kernel5ActorDistributionNetwork/EncodingNetwork/dense_2/biasLActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/kernelJActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/bias*
Tin
2	*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*)
f$R"
 __inference__traced_restore_1951пњ

4
"__inference_get_initial_state_1565

batch_size*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
Т
E
(__inference_function_with_signature_1825
unknown
identityх
PartitionedCallPartitionedCallunknown*
Tin
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*"
fR
__inference_<lambda>_15712
PartitionedCall[
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
: 
К
6
__inference_<lambda>_1571
unknown
identityJ
IdentityIdentityunknown*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
: 
+
р
__inference__traced_save_1915
file_prefixT
Psavev2_actordistributionnetwork_encodingnetwork_dense_kernel_read_readvariableopR
Nsavev2_actordistributionnetwork_encodingnetwork_dense_bias_read_readvariableopV
Rsavev2_actordistributionnetwork_encodingnetwork_dense_1_kernel_read_readvariableopT
Psavev2_actordistributionnetwork_encodingnetwork_dense_1_bias_read_readvariableopV
Rsavev2_actordistributionnetwork_encodingnetwork_dense_2_kernel_read_readvariableopT
Psavev2_actordistributionnetwork_encodingnetwork_dense_2_bias_read_readvariableopk
gsavev2_actordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_kernel_read_readvariableopi
esavev2_actordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_bias_read_readvariableop
savev2_1_const_1

identity_1ЂMergeV2CheckpointsЂSaveV2ЂSaveV2_1
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_05df187366104a44affbfd6ca8300cfa/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameѓ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueћBјB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/6/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/7/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*#
valueBB B B B B B B B 2
SaveV2/shape_and_slicesё
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Psavev2_actordistributionnetwork_encodingnetwork_dense_kernel_read_readvariableopNsavev2_actordistributionnetwork_encodingnetwork_dense_bias_read_readvariableopRsavev2_actordistributionnetwork_encodingnetwork_dense_1_kernel_read_readvariableopPsavev2_actordistributionnetwork_encodingnetwork_dense_1_bias_read_readvariableopRsavev2_actordistributionnetwork_encodingnetwork_dense_2_kernel_read_readvariableopPsavev2_actordistributionnetwork_encodingnetwork_dense_2_bias_read_readvariableopgsavev2_actordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_kernel_read_readvariableopesavev2_actordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_bias_read_readvariableop"/device:CPU:0*
_output_shapes
 *
dtypes

22
SaveV2
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shardЌ
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1Ђ
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slicesб
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const_1^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1у
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЌ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*W
_input_shapesF
D: :d:d:dd:d:dd:d:d:: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:d: 

_output_shapes
:d:$ 

_output_shapes

:dd: 

_output_shapes
:d:$ 

_output_shapes

:dd: 

_output_shapes
:d:$ 

_output_shapes

:d: 

_output_shapes
::	

_output_shapes
: 
ЗЯ
О
__inference_action_1712
	time_step
time_step_1
time_step_2
time_step_3Q
Mactordistributionnetwork_encodingnetwork_dense_matmul_readvariableop_resourceR
Nactordistributionnetwork_encodingnetwork_dense_biasadd_readvariableop_resourceS
Oactordistributionnetwork_encodingnetwork_dense_1_matmul_readvariableop_resourceT
Pactordistributionnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resourceS
Oactordistributionnetwork_encodingnetwork_dense_2_matmul_readvariableop_resourceT
Pactordistributionnetwork_encodingnetwork_dense_2_biasadd_readvariableop_resourceh
dactordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_matmul_readvariableop_resourcei
eactordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_biasadd_readvariableop_resource
identityХ
8ActorDistributionNetwork/EncodingNetwork/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2:
8ActorDistributionNetwork/EncodingNetwork/flatten_3/Const
:ActorDistributionNetwork/EncodingNetwork/flatten_3/ReshapeReshapetime_step_3AActorDistributionNetwork/EncodingNetwork/flatten_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2<
:ActorDistributionNetwork/EncodingNetwork/flatten_3/Reshape
DActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOpMactordistributionnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02F
DActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpН
5ActorDistributionNetwork/EncodingNetwork/dense/MatMulMatMulCActorDistributionNetwork/EncodingNetwork/flatten_3/Reshape:output:0LActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd27
5ActorDistributionNetwork/EncodingNetwork/dense/MatMul
EActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOpNactordistributionnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02G
EActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpН
6ActorDistributionNetwork/EncodingNetwork/dense/BiasAddBiasAdd?ActorDistributionNetwork/EncodingNetwork/dense/MatMul:product:0MActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd28
6ActorDistributionNetwork/EncodingNetwork/dense/BiasAddх
3ActorDistributionNetwork/EncodingNetwork/dense/ReluRelu?ActorDistributionNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџd25
3ActorDistributionNetwork/EncodingNetwork/dense/Relu 
FActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpReadVariableOpOactordistributionnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02H
FActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpС
7ActorDistributionNetwork/EncodingNetwork/dense_1/MatMulMatMulAActorDistributionNetwork/EncodingNetwork/dense/Relu:activations:0NActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd29
7ActorDistributionNetwork/EncodingNetwork/dense_1/MatMul
GActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOpPactordistributionnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02I
GActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpХ
8ActorDistributionNetwork/EncodingNetwork/dense_1/BiasAddBiasAddAActorDistributionNetwork/EncodingNetwork/dense_1/MatMul:product:0OActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd2:
8ActorDistributionNetwork/EncodingNetwork/dense_1/BiasAddы
5ActorDistributionNetwork/EncodingNetwork/dense_1/ReluReluAActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџd27
5ActorDistributionNetwork/EncodingNetwork/dense_1/Relu 
FActorDistributionNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOpReadVariableOpOactordistributionnetwork_encodingnetwork_dense_2_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02H
FActorDistributionNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOpУ
7ActorDistributionNetwork/EncodingNetwork/dense_2/MatMulMatMulCActorDistributionNetwork/EncodingNetwork/dense_1/Relu:activations:0NActorDistributionNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd29
7ActorDistributionNetwork/EncodingNetwork/dense_2/MatMul
GActorDistributionNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOpReadVariableOpPactordistributionnetwork_encodingnetwork_dense_2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02I
GActorDistributionNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOpХ
8ActorDistributionNetwork/EncodingNetwork/dense_2/BiasAddBiasAddAActorDistributionNetwork/EncodingNetwork/dense_2/MatMul:product:0OActorDistributionNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd2:
8ActorDistributionNetwork/EncodingNetwork/dense_2/BiasAddы
5ActorDistributionNetwork/EncodingNetwork/dense_2/ReluReluAActorDistributionNetwork/EncodingNetwork/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџd27
5ActorDistributionNetwork/EncodingNetwork/dense_2/Reluп
[ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul/ReadVariableOpReadVariableOpdactordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02]
[ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul/ReadVariableOp
LActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMulMatMulCActorDistributionNetwork/EncodingNetwork/dense_2/Relu:activations:0cActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2N
LActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMulо
\ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd/ReadVariableOpReadVariableOpeactordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02^
\ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd/ReadVariableOp
MActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAddBiasAddVActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul:product:0dActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2O
MActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAddК
:ActorDistributionNetwork/TanhNormalProjectionNetwork/ConstConst*
_output_shapes
: *
dtype0*
value	B :2<
:ActorDistributionNetwork/TanhNormalProjectionNetwork/Constз
DActorDistributionNetwork/TanhNormalProjectionNetwork/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2F
DActorDistributionNetwork/TanhNormalProjectionNetwork/split/split_dimў
:ActorDistributionNetwork/TanhNormalProjectionNetwork/splitSplitMActorDistributionNetwork/TanhNormalProjectionNetwork/split/split_dim:output:0VActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd:output:0*
T0*:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*
	num_split2<
:ActorDistributionNetwork/TanhNormalProjectionNetwork/splitй
BActorDistributionNetwork/TanhNormalProjectionNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2D
BActorDistributionNetwork/TanhNormalProjectionNetwork/Reshape/shapeЫ
<ActorDistributionNetwork/TanhNormalProjectionNetwork/ReshapeReshapeCActorDistributionNetwork/TanhNormalProjectionNetwork/split:output:0KActorDistributionNetwork/TanhNormalProjectionNetwork/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2>
<ActorDistributionNetwork/TanhNormalProjectionNetwork/Reshapeђ
8ActorDistributionNetwork/TanhNormalProjectionNetwork/ExpExpCActorDistributionNetwork/TanhNormalProjectionNetwork/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2:
8ActorDistributionNetwork/TanhNormalProjectionNetwork/Expѓ
ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/range_dimension_tensor/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/range_dimension_tensor/ConstА
qActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/event_shapeConst*
_output_shapes
:*
dtype0*
valueB:2s
qActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/event_shapeе
БActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/ShapeShape<ActorDistributionNetwork/TanhNormalProjectionNetwork/Exp:y:0*
T0*
_output_shapes
:2Д
БActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shapeи
ПActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2Т
ПActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stackг
СActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2Ф
СActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1г
СActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Ф
СActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2ё	
ЙActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_sliceStridedSliceКActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0ШActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack:output:0ЪActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1:output:0ЪActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2М
ЙActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_sliceј
ЛActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1PackТActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice:output:0*
N*
T0*
_output_shapes
:2О
ЛActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1З
ЗActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2К
ЗActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axisь
ВActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concatConcatV2КActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0ФActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1:output:0РActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axis:output:0*
N*
T0*
_output_shapes
:2Е
ВActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat
ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2Ђ
ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack
ЁActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ўџџџџџџџџ2Є
ЁActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1
ЁActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Є
ЁActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2а
ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_sliceStridedSliceЛActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat:output:0ЈActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack:output:0ЊActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1:output:0ЊActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_sliceЯ
kActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ShapeShapeEActorDistributionNetwork/TanhNormalProjectionNetwork/Reshape:output:0*
T0*
_output_shapes
:2m
kActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ShapeР
yActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2{
yActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stackЭ
{ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2}
{ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1Ф
{ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2}
{ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2Ч
sActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_sliceStridedSlicetActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/Shape:output:0ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack:output:0ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1:output:0ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2u
sActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_sliceљ
ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/prefer_static_broadcast_shape/BroadcastArgsBroadcastArgsЂActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice:output:0|ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice:output:0*
_output_shapes
:2
ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/prefer_static_broadcast_shape/BroadcastArgsы
QActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2S
QActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zerosщ
PActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/onesConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2R
PActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/onesц
PActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zeroConst*
_output_shapes
: *
dtype0*
value	B : 2R
PActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zeroщ
QActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/emptyConst*
_output_shapes
: *
dtype0*
valueB 2S
QActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/emptyУ
;ActorDistributionNetwork/TanhNormalProjectionNetwork/Cast/xConst*
_output_shapes
: *
dtype0*
valueB 2      р?2=
;ActorDistributionNetwork/TanhNormalProjectionNetwork/Cast/xє
9ActorDistributionNetwork/TanhNormalProjectionNetwork/CastCastDActorDistributionNetwork/TanhNormalProjectionNetwork/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 2;
9ActorDistributionNetwork/TanhNormalProjectionNetwork/CastЧ
=ActorDistributionNetwork/TanhNormalProjectionNetwork/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB 2      р?2?
=ActorDistributionNetwork/TanhNormalProjectionNetwork/Cast_1/xњ
;ActorDistributionNetwork/TanhNormalProjectionNetwork/Cast_1CastFActorDistributionNetwork/TanhNormalProjectionNetwork/Cast_1/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 2=
;ActorDistributionNetwork/TanhNormalProjectionNetwork/Cast_1Э
ТActorDistributionNetwork/TanhNormalProjectionNetwork/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/zeroConst*
_output_shapes
: *
dtype0*
value	B : 2Х
ТActorDistributionNetwork/TanhNormalProjectionNetwork/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/zeroа
УActorDistributionNetwork/TanhNormalProjectionNetwork/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/emptyConst*
_output_shapes
: *
dtype0*
valueB 2Ц
УActorDistributionNetwork/TanhNormalProjectionNetwork/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/emptyь
бActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 2д
бActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/sample_shapeЉ
АActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/is_scalar_batch/is_scalar_batchConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2Г
АActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/is_scalar_batch/is_scalar_batchѓ
еActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/pick_vector/condConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2и
еActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/pick_vector/cond
нActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/pick_vector/false_vectorConst*
_output_shapes
:*
dtype0*
valueB:2р
нActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/pick_vector/false_vectorї
зActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/pick_vector_1/condConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2к
зActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/pick_vector_1/cond
оActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/pick_vector_1/true_vectorConst*
_output_shapes
:*
dtype0*
valueB:2с
оActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/pick_vector_1/true_vector
ІActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:2Љ
ІActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/sample_shapeс
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/is_scalar_batch/is_scalar_batchConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/is_scalar_batch/is_scalar_batch
ЊActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/pick_vector/condConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2­
ЊActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/pick_vector/condЕ
ВActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/pick_vector/false_vectorConst*
_output_shapes
:*
dtype0*
valueB:2Е
ВActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/pick_vector/false_vectorЁ
ЌActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/pick_vector_1/condConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2Џ
ЌActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/pick_vector_1/condЗ
ГActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/pick_vector_1/true_vectorConst*
_output_shapes
:*
dtype0*
valueB:2Ж
ГActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/pick_vector_1/true_vector
ЅActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Ј
ЅActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/concat/axisс
 ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/concatConcatV2ЛActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/pick_vector/false_vector:output:0ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/prefer_static_broadcast_shape/BroadcastArgs:r0:0zActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/event_shape:output:0ZActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/empty:output:0ЎActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2Ѓ
 ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/concatЧ
ћActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2ў
ћActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/Constз
њActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/ProdProdЉActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/concat:output:0ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/Const:output:0*
T0*
_output_shapes
: 2§
њActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/ProdЭ
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/concat/values_0PackActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/Prod:output:0*
N*
T0*
_output_shapes
:2
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/concat/values_0д
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/concat/values_1Const*
_output_shapes
: *
dtype0*
valueB 2
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/concat/values_1Ы
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/concat/axisш
ќActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/concatConcatV2ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/concat/values_0:output:0ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/concat/values_1:output:0ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2џ
ќActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/concatм
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/random_normal/meanр
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/random_normal/stddev
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/random_normal/RandomStandardNormalRandomStandardNormalActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/concat:output:0*
T0*#
_output_shapes
:џџџџџџџџџ*
dtype02
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/random_normal/RandomStandardNormal
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/random_normal/mulMulЁActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/random_normal/RandomStandardNormal:output:0ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/random_normal/stddev:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/random_normal/mulф
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/random_normalAddActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/random_normal/mul:z:0ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/random_normal/mean:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/random_normal
љActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/mulMulActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/random_normal:z:0YActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/ones:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2ќ
љActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/mul
љActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/addAddV2§ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/mul:z:0ZActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zeros:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2ќ
љActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/addЋ
ћActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/ShapeShape§ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/add:z:0*
T0*
_output_shapes
:2ў
ћActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/Shapeу
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/strided_slice/stackч
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/strided_slice/stack_1ч
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/strided_slice/stack_2Ї
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/strided_sliceStridedSliceActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/Shape:output:0ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/strided_slice/stack:output:0ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/strided_slice/stack_1:output:0ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/strided_sliceЯ
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/concat_1/axis
ўActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/concat_1ConcatV2ЉActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/concat:output:0ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/strided_slice:output:0ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
ўActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/concat_1Ь
§ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/ReshapeReshape§ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/add:z:0ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
§ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/Reshapeќ

ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ShapeShapeActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/Reshape:output:0*
T0*
_output_shapes
:2Ђ
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ShapeЋ
­ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2А
­ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/strided_slice/stackЏ
ЏActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2В
ЏActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/strided_slice/stack_1Џ
ЏActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2В
ЏActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/strided_slice/stack_2
ЇActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/strided_sliceStridedSliceЈActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/Shape:output:0ЖActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/strided_slice/stack:output:0ИActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/strided_slice/stack_1:output:0ИActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2Њ
ЇActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/strided_slice
ЇActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Њ
ЇActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/concat_1/axis
ЂActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/concat_1ConcatV2ЏActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/sample_shape:output:0АActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/strided_slice:output:0АActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2Ѕ
ЂActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/concat_1С
ЁActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ReshapeReshapeActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal_1/sample/Reshape:output:0ЋActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2Є
ЁActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ReshapeЕ
УActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/LinearOperatorDiag/matvec/mulMul<ActorDistributionNetwork/TanhNormalProjectionNetwork/Exp:y:0ЊActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/Reshape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2Ц
УActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/LinearOperatorDiag/matvec/mulћ
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift/forward/addAddV2ЧActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/LinearOperatorDiag/matvec/mul:z:0EActorDistributionNetwork/TanhNormalProjectionNetwork/Reshape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift/forward/addт
ЪActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ShapeShapeActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift/forward/add:z:0*
T0*
_output_shapes
:2Э
ЪActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/Shape
иActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2л
иActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack
кActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2н
кActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_1
кActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2н
кActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_2
вActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_sliceStridedSliceгActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/Shape:output:0сActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack:output:0уActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_1:output:0уActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2е
вActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_sliceщ
аActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2г
аActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat/axisю
ЫActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concatConcatV2кActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/sample_shape:output:0лActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice:output:0йActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2Ю
ЫActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concatЬ
ЬActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ReshapeReshapeActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_1/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift/forward/add:z:0дActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2Я
ЬActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ReshapeЛ
бActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/TanhTanhеActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/Reshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2д
бActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/Tanhй
оActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar/forward/IdentityIdentityеActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2с
оActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar/forward/Identity
йActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar/forward/mulMulчActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar/forward/Identity:output:0?ActorDistributionNetwork/TanhNormalProjectionNetwork/Cast_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2м
йActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar/forward/mul
йActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar/forward/addAddV2нActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar/forward/mul:z:0=ActorDistributionNetwork/TanhNormalProjectionNetwork/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2м
йActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar/forward/addw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
clip_by_value/Minimum/yэ
clip_by_value/MinimumMinimumнActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_affine_scalar/forward/add:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
clip_by_valuee
IdentityIdentityclip_by_value:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*w
_input_shapesf
d:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:::::::::N J
#
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	time_step:NJ
#
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	time_step:NJ
#
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	time_step:VR
+
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	time_step:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: 
Ы
?
"__inference_signature_wrapper_1833
unknown
identityє
PartitionedCallPartitionedCallunknown*
Tin
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*1
f,R*
(__inference_function_with_signature_18252
PartitionedCall[
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
: 
Э

(__inference_function_with_signature_1780
	step_type

reward
discount
observation
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityЂStatefulPartitionedCallЇ
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ**
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*/
f*R(
&__inference_polymorphic_action_fn_17612
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*w
_input_shapesf
d:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
#
_output_shapes
:џџџџџџџџџ
%
_user_specified_name0/step_type:MI
#
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
0/reward:OK
#
_output_shapes
:џџџџџџџџџ
$
_user_specified_name
0/discount:ZV
+
_output_shapes
:џџџџџџџџџ
'
_user_specified_name0/observation:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: 
Г/
Ђ
 __inference__traced_restore_1951
file_prefixJ
Fassignvariableop_actordistributionnetwork_encodingnetwork_dense_kernelJ
Fassignvariableop_1_actordistributionnetwork_encodingnetwork_dense_biasN
Jassignvariableop_2_actordistributionnetwork_encodingnetwork_dense_1_kernelL
Hassignvariableop_3_actordistributionnetwork_encodingnetwork_dense_1_biasN
Jassignvariableop_4_actordistributionnetwork_encodingnetwork_dense_2_kernelL
Hassignvariableop_5_actordistributionnetwork_encodingnetwork_dense_2_biasc
_assignvariableop_6_actordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_kernela
]assignvariableop_7_actordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_bias

identity_9ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_2ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7Ђ	RestoreV2ЂRestoreV2_1љ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueћBјB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/6/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/7/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*#
valueBB B B B B B B B 2
RestoreV2/shape_and_slicesг
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*4
_output_shapes"
 ::::::::*
dtypes

22
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

IdentityЖ
AssignVariableOpAssignVariableOpFassignvariableop_actordistributionnetwork_encodingnetwork_dense_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1М
AssignVariableOp_1AssignVariableOpFassignvariableop_1_actordistributionnetwork_encodingnetwork_dense_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2Р
AssignVariableOp_2AssignVariableOpJassignvariableop_2_actordistributionnetwork_encodingnetwork_dense_1_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3О
AssignVariableOp_3AssignVariableOpHassignvariableop_3_actordistributionnetwork_encodingnetwork_dense_1_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4Р
AssignVariableOp_4AssignVariableOpJassignvariableop_4_actordistributionnetwork_encodingnetwork_dense_2_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5О
AssignVariableOp_5AssignVariableOpHassignvariableop_5_actordistributionnetwork_encodingnetwork_dense_2_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6е
AssignVariableOp_6AssignVariableOp_assignvariableop_6_actordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7г
AssignVariableOp_7AssignVariableOp]assignvariableop_7_actordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7Ј
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slicesФ
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp

Identity_8Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_8

Identity_9IdentityIdentity_8:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2

Identity_9"!

identity_9Identity_9:output:0*5
_input_shapes$
": ::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ц

&__inference_polymorphic_action_fn_1761
	time_step
time_step_1
time_step_2
time_step_3
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityЂStatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCall	time_steptime_step_1time_step_2time_step_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ**
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8* 
fR
__inference_action_17122
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*w
_input_shapesf
d:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
#
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	time_step:NJ
#
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	time_step:NJ
#
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	time_step:VR
+
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	time_step:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: 
Ќ
Ћ
&__inference_polymorphic_action_fn_1858
time_step_step_type
time_step_reward
time_step_discount
time_step_observation
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityЂStatefulPartitionedCallР
StatefulPartitionedCallStatefulPartitionedCalltime_step_step_typetime_step_rewardtime_step_discounttime_step_observationunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ**
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8* 
fR
__inference_action_17122
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*w
_input_shapesf
d:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
#
_output_shapes
:џџџџџџџџџ
-
_user_specified_nametime_step/step_type:UQ
#
_output_shapes
:џџџџџџџџџ
*
_user_specified_nametime_step/reward:WS
#
_output_shapes
:џџџџџџџџџ
,
_user_specified_nametime_step/discount:b^
+
_output_shapes
:џџџџџџџџџ
/
_user_specified_nametime_step/observation:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: 
Щ
џ
"__inference_signature_wrapper_1806
discount
observation

reward
	step_type
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityЂStatefulPartitionedCallЉ
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ**
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8*1
f,R*
(__inference_function_with_signature_17802
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*w
_input_shapesf
d:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
#
_output_shapes
:џџџџџџџџџ
$
_user_specified_name
0/discount:ZV
+
_output_shapes
:џџџџџџџџџ
'
_user_specified_name0/observation:MI
#
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
0/reward:PL
#
_output_shapes
:џџџџџџџџџ
%
_user_specified_name0/step_type:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: 
Д

&__inference_polymorphic_action_fn_1731
	step_type

reward
discount
observation
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ**
_read_only_resource_inputs

	
**
config_proto

GPU 

CPU2J 8* 
fR
__inference_action_17122
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*w
_input_shapesf
d:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
#
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	step_type:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_namereward:MI
#
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
discount:XT
+
_output_shapes
:џџџџџџџџџ
%
_user_specified_nameobservation:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: 

:
(__inference_function_with_signature_1813

batch_sizeь
PartitionedCallPartitionedCall
batch_size*
Tin
2*

Tout
 *
_output_shapes
 * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*+
f&R$
"__inference_get_initial_state_18122
PartitionedCall*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size

4
"__inference_get_initial_state_1812

batch_size*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size

4
"__inference_signature_wrapper_1818

batch_sizeђ
PartitionedCallPartitionedCall
batch_size*
Tin
2*

Tout
 *
_output_shapes
 * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*1
f,R*
(__inference_function_with_signature_18132
PartitionedCall*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size"ЏL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ч
actionМ
4

0/discount&
action_0/discount:0џџџџџџџџџ
B
0/observation1
action_0/observation:0џџџџџџџџџ
0
0/reward$
action_0/reward:0џџџџџџџџџ
6
0/step_type'
action_0/step_type:0џџџџџџџџџ:
action0
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict*e
get_initial_stateP
2

batch_size$
get_initial_state_batch_size:0 tensorflow/serving/predict*R
get_train_step@"
int32
PartitionedCall_1:0 tensorflow/serving/predict:й

_actor_network
model_variables

signatures

\action
]get_initial_state
^
train_step"
_generic_user_object
а
_encoder
_projection_networks
trainable_variables
regularization_losses
	variables
		keras_api
___call__
*`&call_and_return_all_conditional_losses"
_tf_keras_network§{"class_name": "ActorDistributionNetwork", "name": "ActorDistributionNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"layer was saved without config": true}, "is_graph_network": false}
X

0
1
2
3
4
5
6
7"
trackable_list_wrapper
N

aaction
bget_initial_state
cget_train_step"
signature_map
В
_postprocessing_layers
trainable_variables
regularization_losses
	variables
	keras_api
d__call__
*e&call_and_return_all_conditional_losses"
_tf_keras_networkы{"class_name": "EncodingNetwork", "name": "EncodingNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"layer was saved without config": true}, "is_graph_network": false}
Х
_projection_layer
trainable_variables
regularization_losses
	variables
	keras_api
f__call__
*g&call_and_return_all_conditional_losses"
_tf_keras_network{"class_name": "TanhNormalProjectionNetwork", "name": "TanhNormalProjectionNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"layer was saved without config": true}, "is_graph_network": false}
X

0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
X

0
1
2
3
4
5
6
7"
trackable_list_wrapper
­

layers
non_trainable_variables
trainable_variables
regularization_losses
layer_metrics
	variables
layer_regularization_losses
 metrics
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
G:Ed25ActorDistributionNetwork/EncodingNetwork/dense/kernel
A:?d23ActorDistributionNetwork/EncodingNetwork/dense/bias
I:Gdd27ActorDistributionNetwork/EncodingNetwork/dense_1/kernel
C:Ad25ActorDistributionNetwork/EncodingNetwork/dense_1/bias
I:Gdd27ActorDistributionNetwork/EncodingNetwork/dense_2/kernel
C:Ad25ActorDistributionNetwork/EncodingNetwork/dense_2/bias
^:\d2LActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/kernel
X:V2JActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/bias
<
!0
"1
#2
$3"
trackable_list_wrapper
J

0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
J

0
1
2
3
4
5"
trackable_list_wrapper
­

%layers
&non_trainable_variables
trainable_variables
regularization_losses
'layer_metrics
	variables
(layer_regularization_losses
)metrics
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
п

kernel
bias
*trainable_variables
+regularization_losses
,	variables
-	keras_api
h__call__
*i&call_and_return_all_conditional_losses"К
_tf_keras_layer {"class_name": "Dense", "name": "projection_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "projection_layer", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [0, 100]}}
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­

.layers
/non_trainable_variables
trainable_variables
regularization_losses
0layer_metrics
	variables
1layer_regularization_losses
2metrics
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
У
3trainable_variables
4regularization_losses
5	variables
6	keras_api
j__call__
*k&call_and_return_all_conditional_losses"Д
_tf_keras_layer{"class_name": "Flatten", "name": "flatten_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
л


kernel
bias
7trainable_variables
8regularization_losses
9	variables
:	keras_api
l__call__
*m&call_and_return_all_conditional_losses"Ж
_tf_keras_layer{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 24}}}, "build_input_shape": {"class_name": "TensorShape", "items": [0, 24]}}
с

kernel
bias
;trainable_variables
<regularization_losses
=	variables
>	keras_api
n__call__
*o&call_and_return_all_conditional_losses"М
_tf_keras_layerЂ{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [0, 100]}}
с

kernel
bias
?trainable_variables
@regularization_losses
A	variables
B	keras_api
p__call__
*q&call_and_return_all_conditional_losses"М
_tf_keras_layerЂ{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [0, 100]}}
<
!0
"1
#2
$3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­

Clayers
Dnon_trainable_variables
*trainable_variables
+regularization_losses
Elayer_metrics
,	variables
Flayer_regularization_losses
Gmetrics
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­

Hlayers
Inon_trainable_variables
3trainable_variables
4regularization_losses
Jlayer_metrics
5	variables
Klayer_regularization_losses
Lmetrics
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
.

0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
­

Mlayers
Nnon_trainable_variables
7trainable_variables
8regularization_losses
Olayer_metrics
9	variables
Player_regularization_losses
Qmetrics
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­

Rlayers
Snon_trainable_variables
;trainable_variables
<regularization_losses
Tlayer_metrics
=	variables
Ulayer_regularization_losses
Vmetrics
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­

Wlayers
Xnon_trainable_variables
?trainable_variables
@regularization_losses
Ylayer_metrics
A	variables
Zlayer_regularization_losses
[metrics
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
2
&__inference_polymorphic_action_fn_1858
&__inference_polymorphic_action_fn_1731Б
ЊВІ
FullArgSpec(
args 
j	time_step
jpolicy_state
varargs
 
varkw
 
defaultsЂ
Ђ 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
а2Э
"__inference_get_initial_state_1565І
В
FullArgSpec!
args
jself
j
batch_size
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
__inference_<lambda>_1571
ъ2чф
лВз
FullArgSpecU
argsMJ
jself
jobservations
j	step_type
jnetwork_state

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ъ2чф
лВз
FullArgSpecU
argsMJ
jself
jobservations
j	step_type
jnetwork_state

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
XBV
"__inference_signature_wrapper_1806
0/discount0/observation0/reward0/step_type
4B2
"__inference_signature_wrapper_1818
batch_size
&B$
"__inference_signature_wrapper_1833
ц2ур
зВг
FullArgSpecL
argsDA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults

 
Ђ 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ц2ур
зВг
FullArgSpecL
argsDA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults

 
Ђ 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
д2бЮ
ХВС
FullArgSpec?
args74
jself
jinputs
j
outer_rank

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
д2бЮ
ХВС
FullArgSpec?
args74
jself
jinputs
j
outer_rank

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
	J
Const8
__inference_<lambda>_1571rЂ

Ђ 
Њ " O
"__inference_get_initial_state_1565)"Ђ
Ђ


batch_size 
Њ "Ђ ђ
&__inference_polymorphic_action_fn_1731Ч
тЂо
жЂв
ЪВЦ
TimeStep,
	step_type
	step_typeџџџџџџџџџ&
reward
rewardџџџџџџџџџ*
discount
discountџџџџџџџџџ8
observation)&
observationџџџџџџџџџ
Ђ 
Њ "VВS

PolicyStep*
action 
actionџџџџџџџџџ
stateЂ 
infoЂ 
&__inference_polymorphic_action_fn_1858я
Ђ
ўЂњ
ђВю
TimeStep6
	step_type)&
time_step/step_typeџџџџџџџџџ0
reward&#
time_step/rewardџџџџџџџџџ4
discount(%
time_step/discountџџџџџџџџџB
observation30
time_step/observationџџџџџџџџџ
Ђ 
Њ "VВS

PolicyStep*
action 
actionџџџџџџџџџ
stateЂ 
infoЂ С
"__inference_signature_wrapper_1806
мЂи
Ђ 
аЊЬ
.

0/discount 

0/discountџџџџџџџџџ
<
0/observation+(
0/observationџџџџџџџџџ
*
0/reward
0/rewardџџџџџџџџџ
0
0/step_type!
0/step_typeџџџџџџџџџ"/Њ,
*
action 
actionџџџџџџџџџ]
"__inference_signature_wrapper_181870Ђ-
Ђ 
&Њ#
!

batch_size

batch_size "Њ V
"__inference_signature_wrapper_18330rЂ

Ђ 
Њ "Њ

int32
int32 