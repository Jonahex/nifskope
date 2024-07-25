#version 130

uniform sampler2D BaseMap;
uniform sampler2D NormalMap;
uniform sampler2D GlowMap;
uniform sampler2D LightMask;
uniform sampler2D BacklightMap;

uniform bool hasGlowMap;
uniform vec3 glowColor;
uniform float glowMult;

uniform vec3 specColor;
uniform float specStrength;
uniform float specGlossiness;

uniform vec2 uvScale;
uniform vec2 uvOffset;

uniform bool hasSoftlight;
uniform bool hasBacklight;
uniform bool hasRimlight;

uniform float lightingEffect1;
uniform float lightingEffect2;

in vec3 LightDir;
in vec3 ViewDir;

in vec4 ColorEA;
in vec4 ColorD;
in float toneMapScale;

vec3 tonemap(vec3 x)
{
	float a = 0.15;
	float b = 0.50;
	float c = 0.10;
	float d = 0.20;
	float e = 0.02;
	float f = 0.30;

	vec3 z = x * (toneMapScale * 4.22978723);
	z = (z * (a * z + b * c) + d * e) / (z * (a * z + b) + d * f) - e / f;
	return z / (toneMapScale * 0.93333333);
}

void main( void )
{
	vec2 offset = gl_TexCoord[0].st * uvScale + uvOffset;

	vec4 baseMap = texture2D( BaseMap, offset );
	vec4 nmap = texture2D( NormalMap, offset );

	vec4 color;
	color.rgb = baseMap.rgb;

	vec3 normal = normalize(nmap.rgb * 2.0 - 1.0);

	float spec = 0.0;

	vec3 emissive = texture2D( GlowMap, offset ).rgb;

	// Skyrim
	if ( hasGlowMap ) {
		color.rgb += baseMap.rgb * emissive.rgb * glowColor * glowMult;
	}

	vec3 L = normalize(LightDir);
	vec3 E = normalize(ViewDir);
	vec3 R = reflect(-L, normal);
	//vec3 H = normalize( L + E );
	float NdotL = max(dot(normal, L), 0.0);

	color.rgb *= ColorEA.rgb + ColorD.rgb * NdotL;
	color.a = ColorD.a * baseMap.a;

	if ( NdotL > 0.0 && specStrength > 0.0 ) {
		float RdotE = max( dot( R, E ), 0.0 );

		// TODO: Attenuation?

		if ( RdotE > 0.0 ) {
			spec = nmap.a * gl_LightSource[0].specular.r * specStrength * pow(RdotE, 0.8*specGlossiness);
			color.rgb += spec * specColor;
		}
	}

	vec3 backlight;
	if ( hasBacklight ) {
		backlight = texture2D( BacklightMap, offset ).rgb;
		color.rgb += baseMap.rgb * backlight * (1.0 - NdotL) * 0.66;
	}

	vec4 mask;
	if ( hasRimlight || hasSoftlight ) {
		mask = texture2D( LightMask, offset );
	}

	float facing = dot(-L, E);

	vec3 rim;
	if ( hasRimlight ) {
		rim = vec3(( 1.0 - NdotL ) * ( 1.0 - max(dot(normal, E), 0.0)));
		//rim = smoothstep( 0.0, 1.0, rim );
		rim = mask.rgb * pow(rim, vec3(lightingEffect2)) * gl_LightSource[0].diffuse.rgb * vec3(0.66);
		rim *= smoothstep( -0.5, 1.0, facing );
		color.rgb += rim;
	}

	float wrap = dot(normal, -L);

	vec3 soft;
	if ( hasSoftlight ) {
		soft = vec3((1.0 - wrap) * (1.0 - NdotL));
		soft = smoothstep( -1.0, 1.0, soft );

		// TODO: Very approximate, kind of arbitrary. There is surely a more correct way.
		soft *= mask.rgb * pow(soft, vec3(4.0/(lightingEffect1*lightingEffect1))); // * gl_LightSource[0].ambient.rgb;
		//soft *= smoothstep( -1.0, 0.0, soft );
		//soft = mix( soft, color.rgb, gl_LightSource[0].ambient.rgb );

		//soft = smoothstep( 0.0, 1.0, soft );
		soft *= gl_LightSource[0].diffuse.rgb * gl_LightSource[0].ambient.rgb + (0.01 * lightingEffect1*lightingEffect1);

		//soft = clamp( soft, 0.0, 0.5 );
		//soft *= smoothstep( -0.5, 1.0, facing );
		//soft = mix( soft, color.rgb, gl_LightSource[0].diffuse.rgb );
		color.rgb += baseMap.rgb * soft;
	}

	//color = min( color, 1.0 );

	color.rgb = tonemap( color.rgb );
	gl_FragColor = color;
}