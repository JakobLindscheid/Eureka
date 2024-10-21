CMD="$*"
if CMD=""; then
    full_command="udocker run -v ~/Eureka:/workspace/Eureka -w /workspace/Eureka/ --env='GROQ_KEY=$GROQ_KEY' --env='OPENAI_API_KEY=$OPENAI_API_KEY' eureka /bin/bash"
else
    full_command="udocker run -v ~/Eureka:/workspace/Eureka -w /workspace/Eureka/ --env='GROQ_KEY=$GROQ_KEY' --env='OPENAI_API_KEY=$OPENAI_API_KEY' eureka /bin/bash -c '$CMD'"
fi
eval $full_command