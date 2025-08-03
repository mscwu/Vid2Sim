Shader "Custom/ScreenSpaceDepthShader"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _FarPlane ("Far Plane", Float) = 3.0
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            
            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            sampler2D _MainTex;
            sampler2D _CameraDepthTexture;
            float4 _MainTex_ST;
            float _FarPlane;
            
            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                return o;
            }
            
            fixed4 frag (v2f i) : SV_Target
            {
                float depth = SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, i.uv);
                float linearDepth = LinearEyeDepth(depth);
                
                // Normalize depth to [0, 1] range based on far plane
                float normalizedDepth = linearDepth / _FarPlane;
                
                // Output depth to all channels for maximum precision
                return float4(normalizedDepth, normalizedDepth, normalizedDepth, 1);
            }
            ENDCG
        }
    }
}