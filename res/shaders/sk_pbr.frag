/*
 * Copyright (C) 2024 Ilya Perapechka
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#version 130

uniform sampler2D BaseMap;
uniform sampler2D NormalMap;
uniform sampler2D GlowMap;
uniform sampler2D HeightMap;
uniform sampler2D LightMask;
uniform sampler2D BacklightMap;
uniform sampler2D EnvironmentMap;
uniform samplerCube CubeMap;

uniform vec3 subsurfaceColor;
uniform float thickness;

uniform bool hasGlowMap;
uniform vec3 glowColor;
uniform float glowMult;

uniform float alpha;

uniform vec3 tintColor;

uniform bool hasHeightMap;
uniform vec2 uvScale;
uniform vec2 uvOffset;

uniform bool hasEmit;
uniform bool hasSoftlight;
uniform bool hasBacklight;
uniform bool hasRimlight;
uniform bool hasTintColor;
uniform bool hasCubeMap;
uniform bool hasEnvMask;

uniform float lightingEffect1;
uniform float lightingEffect2;

uniform float envReflection;

uniform mat4 worldMatrix;

uniform float specularLevel;
uniform float roughnessScale;
uniform float displacementScale;

varying vec3 LightDir;
varying vec3 ViewDir;

varying vec4 A;
varying vec4 C;
varying vec4 D;

varying mat3 tbnMatrix;

mat3 tbnMatrix_norm = mat3(normalize(tbnMatrix[0]), normalize(tbnMatrix[1]), normalize(tbnMatrix[2]));


vec3 tonemap(vec3 x, float y)
{
	float a = 0.15;
	float b = 0.50;
	float c = 0.10;
	float d = 0.20;
	float e = 0.02;
	float f = 0.30;

	vec3 z = x * (y * 4.22978723);
	z = (z * (a * z + b * c) + d * e) / (z * (a * z + b) + d * f) - e / f;
	return z / (y * 0.93333333);
}

vec3 toGrayscale(vec3 color)
{
	return vec3(dot(vec3(0.3, 0.59, 0.11), color));
}

vec3 getFresnelFactorSchlick(vec3 specularColor, float VdotH)
{
	float Fc = pow(1 - VdotH, 5);  // 1 sub, 3 mul
	return clamp(50.0 * specularColor.g, 0, 1) * Fc + (1 - Fc) * specularColor;
}

float getGeometryFunctionSmithJointApprox(float roughness, float NdotV, float NdotL)
{
	float a = roughness * roughness;
	float Vis_SmithV = NdotL * (NdotV * (1 - a) + a);
	float Vis_SmithL = NdotV * (NdotL * (1 - a) + a);
	return 0.5 / (Vis_SmithV + Vis_SmithL);
}

float getDistributionFunctionGGX(float roughness, float NdotH)
{
	float a2 = pow(roughness, 4);
	float d = max((NdotH * a2 - NdotH) * NdotH + 1, 1e-5);
	return a2 / (d * d);
}

vec3 getSpecularDirectLightMultiplierMicrofacet(float roughness, vec3 specularColor, float NdotL, float NdotV, float NdotH, float VdotH)
{
	float D = getDistributionFunctionGGX(roughness, NdotH);
	float G = getGeometryFunctionSmithJointApprox(roughness, NdotV, NdotL);
	vec3 F = getFresnelFactorSchlick(specularColor, VdotH);

	return D * G * F;
}

vec3 sRGB2Lin(vec3 color)
{
		return pow(color, vec3(2.2, 2.2, 2.2));
}

vec3 Lin2sRGB(vec3 color)
{
		return pow(color, vec3(0.42, 0.42, 0.42));
}

vec2 getEnvBRDFApproxLazarov(float roughness, float NdotV)
{
	const vec4 c0 = vec4(-1, -0.0275, -0.572, 0.022);
	const vec4 c1 = vec4(1, 0.0425, 1.04, -0.04);
	vec4 r = roughness * c0 + c1;
	float a004 = min(r.x * r.x, exp2(-9.28 * NdotV)) * r.x + r.y;
	vec2 AB = vec2(-1.04, 1.04) * a004 + r.zw;
	return AB;
}

float GetMipLevel(vec2 coords)
{
	// Compute the current gradients:
	ivec2 textureDims = textureSize(HeightMap, 0);

	vec2 texCoordsPerSize = coords * textureDims;

	vec2 dxSize = dFdx(texCoordsPerSize);
	vec2 dySize = dFdy(texCoordsPerSize);

	// Find min of change in u and v across quad: compute du and dv magnitude across quad
	vec2 dTexCoords = dxSize * dxSize + dySize * dySize;

	// Standard mipmapping uses max here
	float minTexCoordDelta = max(dTexCoords.x, dTexCoords.y);

	// Compute the current mip level  (* 0.5 is effectively computing a square root before )
	float mipLevel = max(0.5 * log2(minTexCoordDelta), 0);

	return mipLevel;
}

vec2 GetParallaxCoords(float distance, vec2 coords, float mipLevel, vec3 viewDir)
{
	vec3 viewDirTS = normalize(viewDir * tbnMatrix_norm);
	viewDirTS.xy /= viewDirTS.z * 0.7 + 0.3;  // Fix for objects at extreme viewing angles

	float nearBlendToFar = clamp(distance / 2048.0, 0.0, 1.0);
	float maxHeight = 0.1 * displacementScale;
	float minHeight = maxHeight * 0.5;

	if (nearBlendToFar < 1.0) {
		uint numSteps = uint((32 * (1.0 - nearBlendToFar)) + 0.5);
		numSteps = clamp((numSteps + 3u) & ~0x03u, 4u, 32u);

		float stepSize = 1.0 / numSteps;

		vec2 offsetPerStep = viewDirTS.xy * vec2(maxHeight, maxHeight) * stepSize;
		vec2 prevOffset = viewDirTS.xy * vec2(minHeight, minHeight) + coords.xy;

		float prevBound = 1.0;
		float prevHeight = 1.0;

		vec2 pt1 = vec2(0.0, 0.0);
		vec2 pt2 = vec2(0.0, 0.0);

		while (numSteps > 0u)
		{
			vec4 currentOffset[2];
			currentOffset[0] = prevOffset.xyxy - vec4(1, 1, 2, 2) * offsetPerStep.xyxy;
			currentOffset[1] = prevOffset.xyxy - vec4(3, 3, 4, 4) * offsetPerStep.xyxy;
			vec4 currentBound = prevBound - vec4(1, 2, 3, 4) * stepSize;

			vec4 currHeight;
			currHeight.x = textureLod(HeightMap, currentOffset[0].xy, mipLevel).r;
			currHeight.y = textureLod(HeightMap, currentOffset[0].zw, mipLevel).r;
			currHeight.z = textureLod(HeightMap, currentOffset[1].xy, mipLevel).r;
			currHeight.w = textureLod(HeightMap, currentOffset[1].zw, mipLevel).r;

			bvec4 testResult = bvec4(currHeight.x >= currentBound.x, currHeight.y >= currentBound.y, currHeight.z >= currentBound.z, currHeight.w >= currentBound.w);
			if (any(testResult))
			{
				if (testResult.w)
				{
					pt1 = vec2(currentBound.w, currHeight.w);
					pt2 = vec2(currentBound.z, currHeight.z);
				}
				if (testResult.z)
				{
					pt1 = vec2(currentBound.z, currHeight.z);
					pt2 = vec2(currentBound.y, currHeight.y);
				}
				if (testResult.y)
				{
					pt1 = vec2(currentBound.y, currHeight.y);
					pt2 = vec2(currentBound.x, currHeight.x);
				}
				if (testResult.x)
				{
					pt1 = vec2(currentBound.x, currHeight.x);
					pt2 = vec2(prevBound, prevHeight);
				}
				break;
			}

			prevOffset = currentOffset[1].zw;
			prevBound = currentBound.w;
			prevHeight = currHeight.w;
			numSteps -= 4u;
		}

		float delta2 = pt2.x - pt2.y;
		float delta1 = pt1.x - pt1.y;
		float denominator = delta2 - delta1;

		float parallaxAmount = 0.0;
		if (denominator == 0.0)
		{
			parallaxAmount = 0.0;
		}
		else
		{
			parallaxAmount = (pt1.x * delta2 - pt2.x * delta1) / denominator;
		}

		nearBlendToFar *= nearBlendToFar;

		float offset = (1.0 - parallaxAmount) * -maxHeight + minHeight;
		return mix(viewDirTS.xy * offset + coords.xy, coords, nearBlendToFar);
	}

	return coords;
}

void main( void )
{
	vec2 offset = gl_TexCoord[0].st * uvScale + uvOffset;
	
	vec3 V = normalize(ViewDir);
	
	if ( hasHeightMap ) {
		float mipLevel = GetMipLevel(offset);
		offset = GetParallaxCoords(gl_FragDepth, offset, mipLevel, ViewDir);
	}

	vec4 baseMap = texture2D( BaseMap, offset );
	vec4 normalMap = texture2D( NormalMap, offset );
	vec4 glowMap = texture2D( GlowMap, offset );
	vec4 rmaosMap = texture2D( EnvironmentMap, offset );
	
	float roughness = rmaosMap.r * roughnessScale;
	float metallic = rmaosMap.g;
	float ao = rmaosMap.b;
	float reflectance = rmaosMap.a * specularLevel;
	
	vec3 baseColor = baseMap.rgb * C.rgb * (1 - metallic);
	vec3 f0 = mix(vec3(reflectance, reflectance, reflectance), baseMap.rgb, metallic);
	
	vec3 normal = normalize(tbnMatrix_norm * (normalMap.rgb * 2.0 - 1.0));
	if ( !gl_FrontFacing )
		normal *= -1.0;
	
	vec3 L = normalize(LightDir);
	vec3 R = reflect(-L, normal);
	vec3 H = normalize( L + V );
	
	float NdotL = max( dot(normal, L), 0.0 );
	float NdotH = max( dot(normal, H), 0.0 );
	float NdotV = max( dot(normal, V), 0.0 );
	float VdotH = max( dot(H, V), 0.0 );
	
	vec3 ambientLight = sRGB2Lin(A.rgb);
	vec3 directLight = sRGB2Lin(D.rgb);

	// Diffuse
	vec3 diffuse = directLight * NdotL + ambientLight * ao;

	// Specular
	vec3 specular = getSpecularDirectLightMultiplierMicrofacet(roughness, f0, NdotL, NdotV, NdotH, VdotH) * directLight * NdotL;
    vec2 specularBRDF = getEnvBRDFApproxLazarov(roughness, NdotV);
	specular += ambientLight * ao * (f0 * specularBRDF.x + specularBRDF.y);

	// Emissive
	vec3 emissive = glowColor * glowMult;
	if ( hasEmit ) {
		emissive *= glowMap.rgb;
	}
	
	vec3 transmission = vec3(0, 0, 0);
	if (hasRimlight)
	{
		vec4 subsurfaceMap = texture2D(BacklightMap, offset);
		vec3 finalSubsurfaceColor = subsurfaceColor * subsurfaceMap.rgb;
		float finalThickness = thickness * subsurfaceMap.a;
		
		const float subsurfacePower = 12.234;
		float forwardScatter = exp2(clamp(-dot(V, L), 0.0, 1.0) * subsurfacePower - subsurfacePower);
		float backScatter = clamp(NdotL * finalThickness + (1.0 - finalThickness), 0.0, 1.0) * 0.5;
		float subsurface = mix(backScatter, 1, forwardScatter) * (1.0 - finalThickness);
		transmission += finalSubsurfaceColor * subsurface * directLight;
	}
	
	vec4 color;
	color.rgb = baseColor.rgb * diffuse + transmission + specular + emissive;
	color.rgb = tonemap( color.rgb * D.a, A.a );
	color.rgb = Lin2sRGB(color.rgb);
	color.a = C.a * baseMap.a;

	gl_FragColor = color;
	gl_FragColor.a *= alpha;
}
