// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "shader_recompiler/backend/spirv/emit_spirv_instructions.h"
#include "shader_recompiler/backend/spirv/spirv_emit_context.h"
#include "shader_recompiler/precise_float_policy.h"

namespace Shader::Backend::SPIRV {

Id PreserveF32MultiplyRounding(EmitContext& ctx, IR::Inst* inst, Id a, Id b) {
    if (!RequiresSeparateMultiplyRounding(inst)) {
        return ctx.OpFMul(ctx.F32[1], a, b);
    }
    // Fma(a, b, signed_zero(a * b)) produces the rounded F32 product as a single operation.
    // Keeping the operation explicit prevents a host compiler from contracting it with the
    // following guest add, without the severe loop performance cost of NoContraction on NVIDIA.
    // Match the product's zero sign so the replacement also preserves signed-zero multiplication.
    const Id sign_bits{ctx.OpBitwiseAnd(
        ctx.U32[1], ctx.OpBitwiseXor(ctx.U32[1], ctx.OpBitcast(ctx.U32[1], a),
                                    ctx.OpBitcast(ctx.U32[1], b)),
        ctx.ConstU32(0x80000000u))};
    const Id signed_zero{ctx.OpBitcast(ctx.F32[1], sign_bits)};
    return ctx.OpFma(ctx.F32[1], a, b, signed_zero);
}

Id EmitFPAbs32(EmitContext& ctx, Id value) {
    return ctx.OpFAbs(ctx.F32[1], value);
}

Id EmitFPAbs64(EmitContext& ctx, Id value) {
    return ctx.OpFAbs(ctx.F64[1], value);
}

Id EmitFPAdd32(EmitContext& ctx, IR::Inst* inst, Id a, Id b) {
    return ctx.OpFAdd(ctx.F32[1], a, b);
}

Id EmitFPAdd64(EmitContext& ctx, IR::Inst* inst, Id a, Id b) {
    return ctx.OpFAdd(ctx.F64[1], a, b);
}

Id EmitFPSub32(EmitContext& ctx, IR::Inst* inst, Id a, Id b) {
    return ctx.OpFSub(ctx.F32[1], a, b);
}

Id EmitFPFma32(EmitContext& ctx, IR::Inst* inst, Id a, Id b, Id c) {
    return ctx.OpFma(ctx.F32[1], a, b, c);
}

Id EmitFPFma64(EmitContext& ctx, IR::Inst* inst, Id a, Id b, Id c) {
    return ctx.OpFma(ctx.F64[1], a, b, c);
}

Id EmitFPMax32(EmitContext& ctx, Id a, Id b) {
    return ctx.OpFMax(ctx.F32[1], a, b);
}

Id EmitFPMax64(EmitContext& ctx, Id a, Id b) {
    return ctx.OpFMax(ctx.F64[1], a, b);
}

Id EmitFPMin32(EmitContext& ctx, Id a, Id b) {
    return ctx.OpFMin(ctx.F32[1], a, b);
}

Id EmitFPMin64(EmitContext& ctx, Id a, Id b) {
    return ctx.OpFMin(ctx.F64[1], a, b);
}

Id EmitFPMinTri32(EmitContext& ctx, Id a, Id b, Id c) {
    if (ctx.profile.supports_trinary_minmax) {
        return ctx.OpFMin3AMD(ctx.F32[1], a, b, c);
    }
    return ctx.OpFMin(ctx.F32[1], a, ctx.OpFMin(ctx.F32[1], b, c));
}

Id EmitFPMaxTri32(EmitContext& ctx, Id a, Id b, Id c) {
    if (ctx.profile.supports_trinary_minmax) {
        return ctx.OpFMax3AMD(ctx.F32[1], a, b, c);
    }
    return ctx.OpFMax(ctx.F32[1], a, ctx.OpFMax(ctx.F32[1], b, c));
}

Id EmitFPMedTri32(EmitContext& ctx, Id a, Id b, Id c) {
    if (ctx.profile.supports_trinary_minmax) {
        return ctx.OpFMid3AMD(ctx.F32[1], a, b, c);
    }
    const Id mmx{ctx.OpFMin(ctx.F32[1], ctx.OpFMax(ctx.F32[1], a, b), c)};
    return ctx.OpFMax(ctx.F32[1], ctx.OpFMin(ctx.F32[1], a, b), mmx);
}

Id EmitFPMul32(EmitContext& ctx, IR::Inst* inst, Id a, Id b) {
    return PreserveF32MultiplyRounding(ctx, inst, a, b);
}

Id EmitFPMul64(EmitContext& ctx, IR::Inst* inst, Id a, Id b) {
    const Id product{ctx.OpFMul(ctx.F64[1], a, b)};
    if (RequiresSeparateMultiplyRounding(inst)) {
        ctx.Decorate(product, spv::Decoration::NoContraction);
    }
    return product;
}

Id EmitFPDiv32(EmitContext& ctx, IR::Inst* inst, Id a, Id b) {
    return ctx.OpFDiv(ctx.F32[1], a, b);
}

Id EmitFPDiv64(EmitContext& ctx, IR::Inst* inst, Id a, Id b) {
    return ctx.OpFDiv(ctx.F64[1], a, b);
}

Id EmitFPNeg32(EmitContext& ctx, Id value) {
    return ctx.OpFNegate(ctx.F32[1], value);
}

Id EmitFPNeg64(EmitContext& ctx, Id value) {
    return ctx.OpFNegate(ctx.F64[1], value);
}

Id EmitFPSin(EmitContext& ctx, Id value) {
    return ctx.OpSin(ctx.F32[1], ctx.OpFMul(ctx.F32[1], ctx.pi_x2, value));
}

Id EmitFPCos(EmitContext& ctx, Id value) {
    return ctx.OpCos(ctx.F32[1], ctx.OpFMul(ctx.F32[1], ctx.pi_x2, value));
}

Id EmitFPExp2(EmitContext& ctx, Id value) {
    return ctx.OpExp2(ctx.F32[1], value);
}

Id EmitFPPow(EmitContext& ctx, Id x, Id y) {
    return ctx.OpPow(ctx.F32[1], x, y);
}

Id EmitFPLdexp(EmitContext& ctx, Id value, Id exp) {
    return ctx.OpLdexp(ctx.F32[1], value, exp);
}

Id EmitFPLog2(EmitContext& ctx, Id value) {
    return ctx.OpLog2(ctx.F32[1], value);
}

Id EmitFPRecip32(EmitContext& ctx, Id value) {
    return ctx.OpFDiv(ctx.F32[1], ctx.ConstF32(1.0f), value);
}

Id EmitFPRecip64(EmitContext& ctx, Id value) {
    return ctx.OpFDiv(ctx.F64[1], ctx.Constant(ctx.F64[1], f64{1.0}), value);
}

Id EmitFPRecipSqrt32(EmitContext& ctx, Id value) {
    return ctx.OpInverseSqrt(ctx.F32[1], value);
}

Id EmitFPRecipSqrt64(EmitContext& ctx, Id value) {
    return ctx.OpInverseSqrt(ctx.F64[1], value);
}

Id EmitFPSqrt(EmitContext& ctx, Id value) {
    return ctx.OpSqrt(ctx.F32[1], value);
}

Id EmitFPSaturate32(EmitContext& ctx, Id value) {
    const Id zero{ctx.ConstF32(f32{0.0})};
    const Id one{ctx.ConstF32(f32{1.0})};
    return ctx.OpFClamp(ctx.F32[1], value, zero, one);
}

Id EmitFPSaturate64(EmitContext& ctx, Id value) {
    const Id zero{ctx.Constant(ctx.F64[1], f64{0.0})};
    const Id one{ctx.Constant(ctx.F64[1], f64{1.0})};
    return ctx.OpFClamp(ctx.F64[1], value, zero, one);
}

Id EmitFPClamp32(EmitContext& ctx, Id value, Id min_value, Id max_value) {
    return ctx.OpFClamp(ctx.F32[1], value, min_value, max_value);
}

Id EmitFPClamp64(EmitContext& ctx, Id value, Id min_value, Id max_value) {
    return ctx.OpFClamp(ctx.F64[1], value, min_value, max_value);
}

Id EmitFPRoundEven32(EmitContext& ctx, Id value) {
    return ctx.OpRoundEven(ctx.F32[1], value);
}

Id EmitFPRoundEven64(EmitContext& ctx, Id value) {
    return ctx.OpRoundEven(ctx.F64[1], value);
}

Id EmitFPFloor32(EmitContext& ctx, Id value) {
    return ctx.OpFloor(ctx.F32[1], value);
}

Id EmitFPFloor64(EmitContext& ctx, Id value) {
    return ctx.OpFloor(ctx.F64[1], value);
}

Id EmitFPCeil32(EmitContext& ctx, Id value) {
    return ctx.OpCeil(ctx.F32[1], value);
}

Id EmitFPCeil64(EmitContext& ctx, Id value) {
    return ctx.OpCeil(ctx.F64[1], value);
}

Id EmitFPTrunc32(EmitContext& ctx, Id value) {
    return ctx.OpTrunc(ctx.F32[1], value);
}

Id EmitFPTrunc64(EmitContext& ctx, Id value) {
    return ctx.OpTrunc(ctx.F64[1], value);
}

Id EmitFPFract32(EmitContext& ctx, Id value) {
    return ctx.OpFract(ctx.F32[1], value);
}

Id EmitFPFract64(EmitContext& ctx, Id value) {
    return ctx.OpFract(ctx.F64[1], value);
}

Id EmitFPFrexpSig32(EmitContext& ctx, Id value) {
    const auto frexp = ctx.OpFrexpStruct(ctx.frexp_result_f32, value);
    return ctx.OpCompositeExtract(ctx.F32[1], frexp, 0);
}

Id EmitFPFrexpSig64(EmitContext& ctx, Id value) {
    const auto frexp = ctx.OpFrexpStruct(ctx.frexp_result_f64, value);
    return ctx.OpCompositeExtract(ctx.F64[1], frexp, 0);
}

Id EmitFPFrexpExp32(EmitContext& ctx, Id value) {
    const auto frexp = ctx.OpFrexpStruct(ctx.frexp_result_f32, value);
    return ctx.OpBitcast(ctx.U32[1], ctx.OpCompositeExtract(ctx.S32[1], frexp, 1));
}

Id EmitFPFrexpExp64(EmitContext& ctx, Id value) {
    const auto frexp = ctx.OpFrexpStruct(ctx.frexp_result_f64, value);
    return ctx.OpBitcast(ctx.U32[1], ctx.OpCompositeExtract(ctx.S32[1], frexp, 1));
}

Id EmitFPOrdEqual16(EmitContext& ctx, Id lhs, Id rhs) {
    return ctx.OpFOrdEqual(ctx.U1[1], lhs, rhs);
}

Id EmitFPOrdEqual32(EmitContext& ctx, Id lhs, Id rhs) {
    return ctx.OpFOrdEqual(ctx.U1[1], lhs, rhs);
}

Id EmitFPOrdEqual64(EmitContext& ctx, Id lhs, Id rhs) {
    return ctx.OpFOrdEqual(ctx.U1[1], lhs, rhs);
}

Id EmitFPUnordEqual16(EmitContext& ctx, Id lhs, Id rhs) {
    return ctx.OpFUnordEqual(ctx.U1[1], lhs, rhs);
}

Id EmitFPUnordEqual32(EmitContext& ctx, Id lhs, Id rhs) {
    return ctx.OpFUnordEqual(ctx.U1[1], lhs, rhs);
}

Id EmitFPUnordEqual64(EmitContext& ctx, Id lhs, Id rhs) {
    return ctx.OpFUnordEqual(ctx.U1[1], lhs, rhs);
}

Id EmitFPOrdNotEqual16(EmitContext& ctx, Id lhs, Id rhs) {
    return ctx.OpFOrdNotEqual(ctx.U1[1], lhs, rhs);
}

Id EmitFPOrdNotEqual32(EmitContext& ctx, Id lhs, Id rhs) {
    return ctx.OpFOrdNotEqual(ctx.U1[1], lhs, rhs);
}

Id EmitFPOrdNotEqual64(EmitContext& ctx, Id lhs, Id rhs) {
    return ctx.OpFOrdNotEqual(ctx.U1[1], lhs, rhs);
}

Id EmitFPUnordNotEqual16(EmitContext& ctx, Id lhs, Id rhs) {
    return ctx.OpFUnordNotEqual(ctx.U1[1], lhs, rhs);
}

Id EmitFPUnordNotEqual32(EmitContext& ctx, Id lhs, Id rhs) {
    return ctx.OpFUnordNotEqual(ctx.U1[1], lhs, rhs);
}

Id EmitFPUnordNotEqual64(EmitContext& ctx, Id lhs, Id rhs) {
    return ctx.OpFUnordNotEqual(ctx.U1[1], lhs, rhs);
}

Id EmitFPOrdLessThan16(EmitContext& ctx, Id lhs, Id rhs) {
    return ctx.OpFOrdLessThan(ctx.U1[1], lhs, rhs);
}

Id EmitFPOrdLessThan32(EmitContext& ctx, Id lhs, Id rhs) {
    return ctx.OpFOrdLessThan(ctx.U1[1], lhs, rhs);
}

Id EmitFPOrdLessThan64(EmitContext& ctx, Id lhs, Id rhs) {
    return ctx.OpFOrdLessThan(ctx.U1[1], lhs, rhs);
}

Id EmitFPUnordLessThan16(EmitContext& ctx, Id lhs, Id rhs) {
    return ctx.OpFUnordLessThan(ctx.U1[1], lhs, rhs);
}

Id EmitFPUnordLessThan32(EmitContext& ctx, Id lhs, Id rhs) {
    return ctx.OpFUnordLessThan(ctx.U1[1], lhs, rhs);
}

Id EmitFPUnordLessThan64(EmitContext& ctx, Id lhs, Id rhs) {
    return ctx.OpFUnordLessThan(ctx.U1[1], lhs, rhs);
}

Id EmitFPOrdGreaterThan16(EmitContext& ctx, Id lhs, Id rhs) {
    return ctx.OpFOrdGreaterThan(ctx.U1[1], lhs, rhs);
}

Id EmitFPOrdGreaterThan32(EmitContext& ctx, Id lhs, Id rhs) {
    return ctx.OpFOrdGreaterThan(ctx.U1[1], lhs, rhs);
}

Id EmitFPOrdGreaterThan64(EmitContext& ctx, Id lhs, Id rhs) {
    return ctx.OpFOrdGreaterThan(ctx.U1[1], lhs, rhs);
}

Id EmitFPUnordGreaterThan16(EmitContext& ctx, Id lhs, Id rhs) {
    return ctx.OpFUnordGreaterThan(ctx.U1[1], lhs, rhs);
}

Id EmitFPUnordGreaterThan32(EmitContext& ctx, Id lhs, Id rhs) {
    return ctx.OpFUnordGreaterThan(ctx.U1[1], lhs, rhs);
}

Id EmitFPUnordGreaterThan64(EmitContext& ctx, Id lhs, Id rhs) {
    return ctx.OpFUnordGreaterThan(ctx.U1[1], lhs, rhs);
}

Id EmitFPOrdLessThanEqual16(EmitContext& ctx, Id lhs, Id rhs) {
    return ctx.OpFOrdLessThanEqual(ctx.U1[1], lhs, rhs);
}

Id EmitFPOrdLessThanEqual32(EmitContext& ctx, Id lhs, Id rhs) {
    return ctx.OpFOrdLessThanEqual(ctx.U1[1], lhs, rhs);
}

Id EmitFPOrdLessThanEqual64(EmitContext& ctx, Id lhs, Id rhs) {
    return ctx.OpFOrdLessThanEqual(ctx.U1[1], lhs, rhs);
}

Id EmitFPUnordLessThanEqual16(EmitContext& ctx, Id lhs, Id rhs) {
    return ctx.OpFUnordLessThanEqual(ctx.U1[1], lhs, rhs);
}

Id EmitFPUnordLessThanEqual32(EmitContext& ctx, Id lhs, Id rhs) {
    return ctx.OpFUnordLessThanEqual(ctx.U1[1], lhs, rhs);
}

Id EmitFPUnordLessThanEqual64(EmitContext& ctx, Id lhs, Id rhs) {
    return ctx.OpFUnordLessThanEqual(ctx.U1[1], lhs, rhs);
}

Id EmitFPOrdGreaterThanEqual16(EmitContext& ctx, Id lhs, Id rhs) {
    return ctx.OpFOrdGreaterThanEqual(ctx.U1[1], lhs, rhs);
}

Id EmitFPOrdGreaterThanEqual32(EmitContext& ctx, Id lhs, Id rhs) {
    return ctx.OpFOrdGreaterThanEqual(ctx.U1[1], lhs, rhs);
}

Id EmitFPOrdGreaterThanEqual64(EmitContext& ctx, Id lhs, Id rhs) {
    return ctx.OpFOrdGreaterThanEqual(ctx.U1[1], lhs, rhs);
}

Id EmitFPUnordGreaterThanEqual16(EmitContext& ctx, Id lhs, Id rhs) {
    return ctx.OpFUnordGreaterThanEqual(ctx.U1[1], lhs, rhs);
}

Id EmitFPUnordGreaterThanEqual32(EmitContext& ctx, Id lhs, Id rhs) {
    return ctx.OpFUnordGreaterThanEqual(ctx.U1[1], lhs, rhs);
}

Id EmitFPUnordGreaterThanEqual64(EmitContext& ctx, Id lhs, Id rhs) {
    return ctx.OpFUnordGreaterThanEqual(ctx.U1[1], lhs, rhs);
}

Id EmitFPIsNan16(EmitContext& ctx, Id value) {
    return ctx.OpIsNan(ctx.U1[1], value);
}

Id EmitFPIsNan32(EmitContext& ctx, Id value) {
    return ctx.OpIsNan(ctx.U1[1], value);
}

Id EmitFPIsNan64(EmitContext& ctx, Id value) {
    return ctx.OpIsNan(ctx.U1[1], value);
}

Id EmitFPIsInf32(EmitContext& ctx, Id value) {
    return ctx.OpIsInf(ctx.U1[1], value);
}

Id EmitFPIsInf64(EmitContext& ctx, Id value) {
    return ctx.OpIsInf(ctx.U1[1], value);
}

Id EmitFPCmpClass32(EmitContext& ctx, Id value, Id class_mask) {
    const Id bits = ctx.OpBitcast(ctx.U32[1], value);
    const Id exponent =
        ctx.OpBitwiseAnd(ctx.U32[1], bits, ctx.ConstU32(0x7f800000u));
    const Id mantissa =
        ctx.OpBitwiseAnd(ctx.U32[1], bits, ctx.ConstU32(0x007fffffu));
    const Id is_negative = ctx.OpINotEqual(
        ctx.U1[1], ctx.OpBitwiseAnd(ctx.U32[1], bits, ctx.ConstU32(0x80000000u)),
        ctx.u32_zero_value);
    const Id exponent_is_zero =
        ctx.OpIEqual(ctx.U1[1], exponent, ctx.u32_zero_value);
    const Id exponent_is_max =
        ctx.OpIEqual(ctx.U1[1], exponent, ctx.ConstU32(0x7f800000u));
    const Id mantissa_is_zero =
        ctx.OpIEqual(ctx.U1[1], mantissa, ctx.u32_zero_value);
    const Id is_quiet_nan = ctx.OpINotEqual(
        ctx.U1[1], ctx.OpBitwiseAnd(ctx.U32[1], mantissa, ctx.ConstU32(0x00400000u)),
        ctx.u32_zero_value);

    const Id nan_class =
        ctx.OpSelect(ctx.U32[1], is_quiet_nan,
                     ctx.ConstU32(static_cast<u32>(IR::FloatClassFunc::QuietNan)),
                     ctx.ConstU32(static_cast<u32>(IR::FloatClassFunc::SignalingNan)));
    const Id infinity_class =
        ctx.OpSelect(ctx.U32[1], is_negative,
                     ctx.ConstU32(static_cast<u32>(IR::FloatClassFunc::NegativeInfinity)),
                     ctx.ConstU32(static_cast<u32>(IR::FloatClassFunc::PositiveInfinity)));
    const Id zero_class =
        ctx.OpSelect(ctx.U32[1], is_negative,
                     ctx.ConstU32(static_cast<u32>(IR::FloatClassFunc::NegativeZero)),
                     ctx.ConstU32(static_cast<u32>(IR::FloatClassFunc::PositiveZero)));
    const Id denorm_class =
        ctx.OpSelect(ctx.U32[1], is_negative,
                     ctx.ConstU32(static_cast<u32>(IR::FloatClassFunc::NegativeDenorm)),
                     ctx.ConstU32(static_cast<u32>(IR::FloatClassFunc::PositiveDenorm)));
    const Id normal_class =
        ctx.OpSelect(ctx.U32[1], is_negative,
                     ctx.ConstU32(static_cast<u32>(IR::FloatClassFunc::NegativeNormal)),
                     ctx.ConstU32(static_cast<u32>(IR::FloatClassFunc::PositiveNormal)));

    const Id finite_zero_or_denorm =
        ctx.OpSelect(ctx.U32[1], mantissa_is_zero, zero_class, denorm_class);
    const Id finite_class =
        ctx.OpSelect(ctx.U32[1], exponent_is_zero, finite_zero_or_denorm, normal_class);
    const Id nan_or_infinity =
        ctx.OpSelect(ctx.U32[1], mantissa_is_zero, infinity_class, nan_class);
    const Id value_class =
        ctx.OpSelect(ctx.U32[1], exponent_is_max, nan_or_infinity, finite_class);
    const Id selected_class = ctx.OpBitwiseAnd(ctx.U32[1], value_class, class_mask);
    return ctx.OpINotEqual(ctx.U1[1], selected_class, ctx.u32_zero_value);
}

} // namespace Shader::Backend::SPIRV
