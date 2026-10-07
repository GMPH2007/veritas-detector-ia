"""
detector/train_model.py
Entrenamiento de la Red Neuronal Forense y Ensamble Calibrado (MLP + Gradient Boosting + Random Forest)
para detección de Inteligencia Artificial (ChatGPT, Gemini, Claude, Perplexity, DeepSeek)
y detección de textos con intentos de evasión / paráfrasis superficial.
"""

import os
import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from detector.features import extract_features

TRAINING_DATA = [
    # =========================================================================
    # CLASE 1: TEXTOS DE INTELIGENCIA ARTIFICIAL (60 MUESTRAS MULTI-MODELO)
    # =========================================================================

    # --- ChatGPT 4o / o1 / GPT-4 (Español) ---
    (
        """En un mundo cada vez más digitalizado, la inteligencia artificial desempeña un papel fundamental
        en la transformación de diversas industrias. Desde la atención médica hasta la educación y el sector
        financiero, estas tecnologías están revolucionando la forma en que interactuamos con la información.
        Cabe destacar que, si bien ofrece ventajas significativas en términos de eficiencia y automatización,
        también plantea importantes desafíos éticos que la sociedad debe abordar. A su vez, es crucial encontrar
        un equilibrio adecuado entre innovación y regulación para asegurar un desarrollo sostenible y equitativo.
        En conclusión, el futuro de la inteligencia artificial dependerá en gran medida de nuestra capacidad
        para guiar su evolución de manera responsable.""",
        1
    ),
    (
        """La fotosíntesis es un proceso biológico fundamental mediante el cual los organismos autótrofos
        convierten la energía lumínica en energía química. Este fenómeno se lleva a cabo en los cloroplastos,
        donde la clorofila absorbe la radiación solar para sintetizar moléculas orgánicas a partir de dióxido
        de carbono y agua. En este sentido, la fase luminosa produce adenosín trifosfato y nicotinamida adenina
        dinucleótido fosfato, compuestos esenciales para el ciclo de Calvin. En última instancia, la fotosíntesis
        desempeña un papel crucial en el mantenimiento del equilibrio ecológico global al oxigenar la atmósfera.""",
        1
    ),
    (
        """La lectura fomenta el pensamiento crítico y enriquece el vocabulario de las personas.
        Adentrarse en las páginas de una buena obra literaria permite explorar diferentes perspectivas
        y desarrollar una profunda empatía hacia experiencias ajenas. Cabe mencionar que en la era de las
        pantallas y la gratificación instantánea, cultivar el hábito de la lectura se convierte en un faro
        de serenidad. En última instancia, un libro no solo transmite conocimiento, sino que también
        construye un puente hacia la introspección personal.""",
        1
    ),
    (
        """El liderazgo efectivo no se basa únicamente en la autoridad jerárquica, sino en la capacidad de
        inspirar y motivar a los equipos de trabajo. En este sentido, la inteligencia emocional y la escucha
        activa son habilidades de vital importancia que todo líder contemporáneo debe cultivar con esmero.
        Asimismo, fomentar un ambiente de confianza y colaboración mutua permite maximizar el potencial de
        cada integrante. En última instancia, el verdadero liderazgo consiste en empoderar a los demás para
        alcanzar objetivos comunes.""",
        1
    ),
    (
        """La biodiversidad de los océanos constituye un tapiz invaluable para el equilibrio ecológico del
        planeta. Los arrecifes de coral, a menudo denominados las selvas tropicales del mar, albergan una
        diversidad asombrosa de especies marinas. No obstante, la acidificación de las aguas y la contaminación
        por plásticos amenazan gravemente su supervivencia. Es crucial destacar la necesidad de implementar
        políticas de conservación marina rigurosas y fomentar una mayor conciencia ciudadana para proteger
        este valioso patrimonio natural.""",
        1
    ),
    (
        """En el panorama económico actual, la gestión del tiempo y la productividad son temas centrales.
        Implementar metodologías ágiles permite a las organizaciones optimizar sus flujos de trabajo y responder
        con rapidez a las demandas del mercado. Vale la pena señalar que la automatización de procesos repetitivos
        libera tiempo valioso para tareas de carácter estratégico e innovador. En conclusión, la adaptabilidad
        organizacional es la clave para la resiliencia en tiempos de incertidumbre.""",
        1
    ),
    (
        """La energía solar fotovoltaica representa una de las alternativas más prometedoras frente a la crisis climática.
        Mediante la captación de radiación solar a través de paneles semiconductores, es posible generar electricidad
        limpia y renovable a gran escala. Es fundamental tener en cuenta que la disminución progresiva de los costos
        de fabricación ha facilitado su adopción en hogares e industrias de todo el planeta. A su vez, el desarrollo
        de sistemas de almacenamiento avanzados permite mitigar la intermitencia del suministro. Para concluir, la transición
        energética resulta indispensable para alcanzar los objetivos de sostenibilidad global.""",
        1
    ),
    (
        """La Revolución Industrial supuso un punto de inflexión decisivo en la historia socioeconómica de la humanidad.
        La invención de la máquina de vapor y la mecanización de los talleres textiles transformaron radicalmente los
        métodos de producción tradicionales. Cabe destacar que este proceso aceleró la urbanización y dio origen a nuevas
        estructuras laborales en las principales metrópolis europeas. Asimismo, impulsó el comercio internacional a niveles
        nunca antes vistos. En resumen, sentó las bases materiales del desarrollo tecnológico e industrial moderno.""",
        1
    ),
    (
        """El aprendizaje automático, o machine learning, es una rama medular de las ciencias computacionales modernas.
        A través del entrenamiento de algoritmos con grandes volúmenes de datos, los sistemas son capaces de identificar
        patrones complejos y realizar predicciones con un elevado grado de precisión. Cabe mencionar que su implementación
        abarca desde sistemas de recomendación en plataformas de streaming hasta diagnósticos médicos de alta complejidad.
        En este sentido, la optimización continua de los modelos matemáticos asegura una mayor robustez analítica.
        En conclusión, constituye un pilar indispensable en la era del procesamiento masivo de datos.""",
        1
    ),
    (
        """La ciberseguridad en el entorno corporativo contemporáneo exige un enfoque proactivo y multidimensional.
        Las amenazas informáticas, como el phishing y el ransomware, evolucionan a un ritmo vertiginoso, comprometiendo
        la integridad de la información confidencial. Es imperativo establecer protocolos rigurosos de autenticación
        multifactor y capacitar de forma continua a los colaboradores de las organizaciones. A su vez, las auditorías
        periódicas permiten mitigar posibles vulnerabilidades en las redes. En última instancia, salvaguardar los activos
        digitales es un requisito ineludible para garantizar la continuidad operativa de cualquier entidad.""",
        1
    ),

    # --- Google Gemini (Español) ---
    (
        """Aquí tienes un desglose detallado de los beneficios de la computación cuántica:
        1. Capacidad de procesamiento exponencial gracias al principio de superposición.
        2. Avances significativos en criptografía y seguridad informática.
        3. Simulación molecular precisa para el descubrimiento acelerado de medicamentos.
        
        En términos generales, la computación cuántica no busca reemplazar a los ordenadores tradicionales,
        sino resolver problemas complejos que actualmente son intratables. Es fundamental señalar que la
        tecnología aún se encuentra en una etapa experimental pero promete transformar el panorama científico.""",
        1
    ),
    (
        """El cambio climático es uno de los mayores desafíos que enfrenta la humanidad en el siglo XXI.
        Para comprender su alcance, es fundamental destacar que las emisiones de gases de efecto invernadero
        han alcanzado niveles históricos debido a la actividad industrial.
        
        A continuación, se presentan los aspectos más relevantes:
        - Aumento global de las temperaturas y deshielo polar.
        - Incremento en la frecuencia de eventos climáticos extremos.
        - Impacto severo en la biodiversidad y la seguridad alimentaria.
        
        En este sentido, la transición hacia energías renovables no solo es deseable, sino indispensable.
        Asimismo, la cooperación internacional juega un papel crucial para cumplir con los acuerdos globales.
        En resumen, mitigar el calentamiento global requiere un compromiso colectivo inmediato.""",
        1
    ),
    (
        """Exploremos las ventajas de la nutrición equilibrada en la vida moderna.
        Mantener una ingesta adecuada de macronutrientes permite optimizar los niveles de energía diarios.
        Puntos clave:
        - Consumo variado de vegetales de hoja verde y frutas de estación.
        - Hidratación constante para regular el metabolismo celular.
        - Reducción paulatina de alimentos ultraprocesados.
        En términos generales, pequeños cambios sostenibles en la dieta producen un impacto duradero en la salud integral.""",
        1
    ),
    (
        """Veamos más de cerca el impacto de la inteligencia artificial en el ámbito educativo.
        Las herramientas pedagógicas asistidas por algoritmos permiten una personalización del aprendizaje sin precedentes.
        Aspectos destacados:
        - Adaptación del ritmo de instrucción a las necesidades individuales de cada estudiante.
        - Retroalimentación instantánea en ejercicios y evaluaciones formativas.
        - Automatización de tareas administrativas para los docentes.
        En resumen, la integración tecnológica redefine la experiencia del aula convencional hacia entornos interactivos.""",
        1
    ),
    (
        """Aquí tienes un resumen ejecutivo sobre la importancia del sueño en el rendimiento cognitivo:
        El descanso adecuado resulta imprescindible para la consolidación de la memoria y la restauración neuronal.
        Factores determinantes:
        1. Duración mínima de 7 a 8 horas continuas en adultos jóvenes.
        2. Regulación del ritmo circadiano mediante horarios estables de descanso.
        3. Disminución de la exposición a pantallas luminosas antes de dormir.
        En resumidas cuentas, priorizar una higiene del sueño adecuada repercute directamente en la productividad diaria.""",
        1
    ),

    # --- Claude / Anthropic (Español) ---
    (
        """Es importante reconocer que el debate sobre el teletrabajo involucra múltiples dimensiones.
        Si bien existen argumentos válidos a favor de la flexibilidad horaria y la reducción de tiempos de traslado,
        también es crucial sopesar los riesgos de aislamiento social y la disolución de los límites laborales.
        Desde una perspectiva equilibrada, las organizaciones deben diseñar modelos híbridos adaptados
        a las necesidades particulares de sus colaboradores. Conviene ser cauteloso antes de asumir conclusiones definitivas.""",
        1
    ),
    (
        """Al analizar el impacto de las redes sociales en la juventud, una perspectiva matizada resulta indispensable.
        Por un lado, facilitan la conexión entre comunidades con intereses compartidos y democratizan la información.
        Por otro lado, la exposición continua a estándares idealizados puede deteriorar la autoestima.
        Es crucial comprender que la alfabetización digital y el acompañamiento familiar son herramientas esenciales.""",
        1
    ),
    (
        """Con respecto a la implementación de sistemas de reconocimiento facial en espacios públicos,
        conviene sopesar rigurosamente los beneficios de seguridad frente a las legítimas preocupaciones de privacidad.
        Si bien optimiza la prevención del delito en nodos de alta concurrencia, el riesgo de falsos positivos
        y la vigilancia masiva despiertan dudas éticas fundadas. Resulta prudente señalar que un marco regulatorio
        robusto debe preceder a cualquier despliegue tecnológico a gran escala.""",
        1
    ),
    (
        """El análisis literario del Siglo de Oro español revela una notable riqueza estilística y temática.
        Desde una perspectiva crítica, autores como Cervantes y Quevedo exploraron las contradicciones humanas
        mediante el contraste entre idealismo y desengaño. Es importante reconocer que sus obras no solo reflejaron
        la realidad de su tiempo, sino que establecieron arquetipos universales vigentes en la literatura contemporánea.""",
        1
    ),

    # --- DeepSeek & Perplexity (Español) ---
    (
        """Según los hallazgos recientes, la transición hacia vehículos eléctricos se ha acelerado notablemente.
        Los estudios sugieren que la densidad energética de las celdas de batería continuará incrementándose
        durante la próxima década. Analicemos paso a paso los factores determinantes: infraestructura de recarga rápida,
        subsidios gubernamentales y reciclaje de materiales críticos. En resumen, la evidencia señala que el transporte
        sostenible dominará el mercado automotriz en el mediano plazo.""",
        1
    ),
    (
        """Analicemos paso a paso el funcionamiento del sistema inmunitario frente a agentes patógenos invasores.
        En primer lugar, la respuesta innata proporciona una barrera física y química inmediata contra bacterias y virus.
        En segundo lugar, los linfocitos T y B orquestan la inmunidad adaptativa mediante la síntesis de anticuerpos específicos.
        Desglosando el razonamiento de manera exhaustiva, la memoria inmunológica garantiza una defensa más veloz ante reinfecciones.
        Los datos indican que la vacunación aprovecha este mecanismo para conferir protección duradera a la población.""",
        1
    ),
    (
        """De acuerdo con las fuentes consultadas, la arquitectura de microservicios ofrece ventajas en escalabilidad.
        Examinando minuciosamente la estructura, cada servicio se ejecuta de forma independiente y se comunica vía APIs REST.
        Los estudios sugieren que esto reduce el acoplamiento y facilita despliegues continuos sin interrupción del servicio.
        Sin embargo, la complejidad operativa en monitoreo distribuido y consistencia de datos requiere herramientas especializadas.
        En base a investigaciones recientes, las organizaciones adoptan Kubernetes para automatizar la orquestación.""",
        1
    ),

    # --- Ensayos Académicos e Informativos Sintéticos de IA (Español) ---
    (
        """La economía circular propone un paradigma transformador frente al modelo productivo lineal tradicional.
        En lugar de extraer, fabricar y desechar, este enfoque persigue mantener los materiales en su ciclo más alto de valor.
        Cabe destacar que la reutilización sistemática y el ecodiseño reducen sustancialmente la huella de carbono industrial.
        Asimismo, abre interesantes oportunidades de negocio en el mercado de materias primas secundarias.
        En última instancia, avanzar hacia la circularidad es un imperativo ético y económico en el mundo moderno.""",
        1
    ),
    (
        """El agua potable es un recurso vital cuya disponibilidad se encuentra amenazada por la sobreexplotación.
        El crecimiento demográfico y la agricultura intensiva han incrementado la presión hídrica en diversas regiones.
        Es importante recordar que la gestión integrada de cuencas y el tratamiento de aguas residuales son estrategias clave.
        A su vez, la inversión en tecnologías de desalinización eficiente ofrece alivio a zonas áridas costeras.
        En conclusión, garantizar el acceso universal al agua potable demanda cooperación transfronteriza y voluntad política.""",
        1
    ),
    (
        """La arquitectura bioclimática busca diseñar edificaciones en armonía con el entorno meteorológico local.
        A través del aprovechamiento pasivo de la luz solar, la ventilación cruzada y el aislamiento térmico natural,
        se logra una drástica reducción del consumo de climatización artificial. Vale la pena señalar que el uso de materiales
        autóctonos disminuye además los costos de transporte y emisiones asociadas. En conclusión, edificar de manera sostenible
        no es solo una tendencia estética, sino una respuesta indispensable al desafío ambiental.""",
        1
    ),
    (
        """La ética en la inteligencia artificial generativa plantea interrogantes urgentes sobre autoría y propiedad intelectual.
        El entrenamiento de modelos fundacionales con extensos corpus de datos protegidos genera legítimas controversias legales.
        Es fundamental establecer pautas transparentes de atribución y compensación para los creadores originales de contenido.
        De igual manera, mitigar los sesgos algorítmicos en la generación de textos e imágenes es prioritario.
        En última instancia, el progreso tecnológico debe subordinarse siempre al respeto por los derechos humanos fundamentales.""",
        1
    ),
    (
        """El desarrollo de la computación en la nube ha transformado drásticamente la infraestructura tecnológica empresarial.
        La posibilidad de escalar recursos de procesamiento y almacenamiento bajo demanda reduce costes de mantenimiento local.
        Cabe mencionar que la adopción de esquemas multinube confiere una mayor resiliencia frente a caídas imprevistas de servicio.
        Por consiguiente, las compañías logran acelerar sus ciclos de innovación con menor riesgo financiero inicial.
        En resumen, el cloud computing es un habilitador neurálgico de la transformación digital contemporánea.""",
        1
    ),
    (
        """La globalización cultural en el siglo XXI genera un dinámico intercambio de expresiones y tradiciones artísticas.
        Si bien favorece la difusión instantánea de contenidos y fomenta la tolerancia entre comunidades diversas,
        también suscita inquietudes en torno a la homogeneización de identidades autóctonas. Es crucial destacar que la salvaguardia
        del patrimonio inmaterial depende de políticas culturales inclusivas que promuevan la diversidad local.
        En última instancia, la convivencia pacífica descansa en el enriquecimiento mutuo de las culturas sin imposiciones.""",
        1
    ),

    # --- Muestras de IA "Humanizada" Superficialmente / Spinbots (Español) ---
    (
        """En la época contemporánea, la tecnología de máquinas pensantes asume una función de alta relevancia
        en la vida cotidiana de las personas. Resulta indispensable comprender que, si bien suministra beneficios
        notables en celeridad y automatización, paralelamente genera dilemas éticos que demandan una prudente reflexión.
        Al mismo tiempo, es primordial conservar la mesura entre progreso y normativa.
        En balance general, el destino de tales herramientas reposará en la prudencia con que sean administradas.""",
        1
    ),
    (
        """Actualmente, el fenómeno del calentamiento global constituye una de las mayores encrucijadas para la civilización.
        Resulta oportuno advertir que las emanaciones de gases contaminantes han alcanzado marcas sin precedentes
        debido a la actividad manufacturera. Visto así, la transición hacia modelos de energía limpia se torna
        indispensable para preservar el equilibrio ecológico. De acuerdo con las fuentes, revertir este deterioro
        requiere una acción mancomunada de escala planetaria.""",
        1
    ),
    (
        """Hoy en día, las plataformas de aprendizaje a distancia proporcionan diversas facilidades para estudiantes
        en múltiples rincones del planeta. Vale decir que la eliminación de trabas geográficas favorece una instrucción
        personalizada y dinámica. No obstante, conviene no pasar por alto que las asimetrías de conectividad representan
        desafíos que exigen atención prioritaria. Como balance general, estas herramientas deben actualizarse
        permanentemente para asegurar una cobertura equitativa.""",
        1
    ),
    (
        """En tiempos presentes, la conservación del medio ambiente ocupa un sitio primordial en las agendas públicas.
        Se torna imprescindible asimilar que los recursos naturales poseen límites finitos que debemos salvaguardar.
        A grandes rasgos, las tácticas de reciclaje y reforestación confieren resultados satisfactorios cuando se coordinan
        adecuadamente. Para resumir las consideraciones anteriores, el porvenir del ecosistema demanda una intervención inmediata.""",
        1
    ),
    (
        """En los días que corren, los mecanismos de dinero digital descentralizado adquieren creciente notoriedad internacional.
        Conviene puntualizar que la metodología de bloques encriptados garantiza transferencias sin la obligada mediación bancaria.
        No obstante aquello, la elevada variabilidad de cotización de estos activos acarrea incertidumbre a los inversores novatos.
        En síntesis conclusiva, el aparato normativo deberá adaptarse con agilidad para proteger el sistema financiero global.""",
        1
    ),
    (
        """La indagación en torno a la inteligencia biológica de los cetáceos arroja datos reveladores para la ciencia marina.
        Resulta digno de mención que los delfines exhiben pautas de comunicación vocal dotadas de notable complejidad sintáctica.
        Por añadidura, su facultad para el auxilio recíproco ilustra conductas de genuina empatía social entre congéneres.
        En resumen estimativo, estos descubrimientos consolidan la urgencia de tutelar sus hábitats oceánicos originarios.""",
        1
    ),

    # --- IA en Inglés (ChatGPT, Gemini, Claude, DeepSeek) ---
    (
        """In today's fast-paced digital world, artificial intelligence plays a crucial role in modern society.
        From healthcare diagnostics to financial modeling, these systems are transforming how we process information.
        It is worth noting that while AI offers immense opportunities, it also presents ethical considerations
        that must be navigated with care. In conclusion, striking a balance between technological advancement
        and responsible governance is paramount for the future of humanity.""",
        1
    ),
    (
        """Renewable energy represents a cornerstone in the global effort to combat climate change.
        Solar and wind technologies have made remarkable strides over the past decade, significantly reducing
        production costs. Moreover, modern energy storage systems allow for more reliable grid integration.
        Furthermore, international collaboration is essential to accelerate this transition. To sum up,
        embracing clean energy is not only an environmental imperative, but also an economic opportunity.""",
        1
    ),
    (
        """The novel stands as a testament to the author's ability to weave complex themes of identity and loss.
        Delving into the protagonist's emotional journey reveals a rich tapestry of human resilience.
        The interplay between light and darkness throughout the narrative serves as a reminder of the fragility
        of our choices. In the realm of contemporary literature, this work underscores the importance of empathy.""",
        1
    ),
    (
        """Here's a breakdown of the key factors influencing urban development:
        - Sustainable transportation infrastructure.
        - Efficient waste management and circular economy initiatives.
        - Creation of accessible green public spaces.
        
        Overall, modern cities must adopt a holistic approach that balances economic growth with environmental
        stewardship and social inclusivity. It is important to remember that community engagement remains vital.""",
        1
    ),
    (
        """While there are valid arguments on both sides regarding artificial general intelligence,
        it is worth being cautious about sensationalist predictions. A nuanced perspective requires examining
        both computational thresholds and architectural bottlenecks. From a broader perspective, multidisciplinary
        oversight will remain essential as capabilities continue to advance.""",
        1
    ),
    (
        """Let's analyze step by step the underlying principles of distributed computing architectures.
        First, decentralization enhances fault tolerance by distributing workloads across independent nodes.
        Second, consensus algorithms like Raft and Paxos maintain data consistency despite network partitions.
        Examining thoroughly the performance trade-offs, network latency often becomes the primary bottleneck.
        In summary, modern microservices leverage container orchestration to optimize operational efficiency.""",
        1
    ),
    (
        """According to recent research findings, cognitive behavioral therapy demonstrates significant efficacy
        in treating anxiety disorders. Clinical trials suggest that structured cognitive restructuring helps patients
        refract negative thought cycles. Furthermore, evidence indicates that integrating mindfulness practices
        enhances long-term emotional regulation. In conclusion, empirical data underscores the necessity of personalized therapy.""",
        1
    ),
    (
        """The transition to remote work has profoundly reshaped contemporary corporate culture across industries.
        Organizations now leverage collaborative cloud suites to maintain operational continuity and team engagement.
        However, it is crucial to recognize that boundary blurring between personal and professional spheres
        can lead to employee burnout. In the grand scheme of organizational management, cultivating intentional
        communication is key to fostering lasting workplace satisfaction.""",
        1
    ),
    (
        """Gene editing technologies such as CRISPR-Cas9 represent a monumental leap forward in biological sciences.
        By enabling precise alterations in genomic sequences, researchers can target genetic conditions at their root.
        Nevertheless, the ethical implications of germline editing demand cautious and comprehensive oversight.
        Striking a delicate balance between therapeutic innovation and moral boundaries remains paramount for global bioethics.""",
        1
    ),
    (
        """Cybersecurity in the era of quantum computing faces unprecedented theoretical challenges.
        Traditional public-key encryption schemes relying on integer factorization may become obsolete in the coming decades.
        Consequently, the development of post-quantum cryptographic standards has gained immense momentum among cryptographers.
        It is worth noting that proactive migration to quantum-resistant algorithms is essential to ensure long-term data security.""",
        1
    ),
    (
        """Para resolver esta ecuación diferencial no homogénea de segundo orden, desglosaremos el método paso a paso.
        En primer lugar, determinamos la solución general de la ecuación diferencial homogénea asociada mediante las raíces
        del polinomio característico. En segundo lugar, aplicamos el método de los coeficientes indeterminados para encontrar
        la solución particular. Analicemos paso a paso cada una de las integraciones algebraicas resultantes.
        En conclusión, la superposición de ambas funciones proporciona la solución analítica completa del sistema.""",
        1
    ),
    (
        """Al reflexionar sobre el papel del arte contemporáneo y sus rupturas estilísticas, conviene ser cauteloso ante juicios apresurados.
        Si bien existen argumentos válidos que señalan una pérdida del virtuosismo técnico tradicional, también es crucial
        reconocer que estas manifestaciones expanden los horizontes del pensamiento crítico y la provocación estética.
        Desde una perspectiva equilibrada, la experimentación conceptual y la tradición material enriquecen dialécticamente el patrimonio cultural.""",
        1
    ),
    (
        """Aquí tienes un desglose detallado de los pilares para optimizar la salud cardiovascular:
        1. Realizar al menos 150 minutos semanales de actividad aeróbica de intensidad moderada.
        2. Mantener una alimentación rica en ácidos grasos omega-3 y baja en sodio.
        3. Monitorear periódicamente los niveles de presión arterial y colesterol sérico.
        En términos generales, la adopción consistente de estos hábitos previene patologías crónicas de manera eficaz.""",
        1
    ),
    (
        """La transición hacia matrices energéticas descarbonizadas representa uno de los mayores hitos tecnológicos de nuestra época.
        La integración masiva de parques eólicos y plantas solares exige una modernización sustancial de las redes de distribución
        eléctrica mediante tecnologías de red inteligente. Cabe destacar que el almacenamiento en baterías de iones de litio y el
        hidrógeno verde juegan un papel crucial para mitigar la variabilidad climática. En este sentido, la colaboración público-privada
        resulta fundamental para canalizar las inversiones necesarias. En última instancia, la sostenibilidad energética garantizará
        la viabilidad económica de las futuras generaciones.""",
        1
    ),
    (
        """La exploración espacial tripulada hacia Marte se perfila como la próxima gran frontera de la civilización contemporánea.
        Los desafíos logísticos, que abarcan desde la protección contra la radiación cósmica hasta el soporte vital en misiones prolongadas,
        demandan soluciones de ingeniería aeroespacial de vanguardia. Vale la pena señalar que el desarrollo de cohetes reutilizables ha
        reducido drásticamente los costes de inserción orbital. A su vez, el establecimiento de bases lunares permanentes servirá como plataforma
        de prueba estratégica. En conclusión, expandir nuestra presencia cósmica enriquecerá tanto nuestro conocimiento científico
        como nuestra visión colectiva del universo.""",
        1
    ),
    (
        """Según los hallazgos recientes en neurociencia cognitiva, el cerebro humano conserva su capacidad de remodelación sináptica
        a lo largo de toda la vida adulta. Los estudios sugieren que la adquisición continuada de nuevas habilidades estimula la angiogénesis
        y la neurogénesis en el hipocampo. La evidencia señala que mantener una estimulación intelectual constante ejerce un efecto neuroprotector
        frente a patologías degenerativas. En base a investigaciones recientes, las intervenciones no farmacológicas representan un pilar
        clave en la preservación de las facultades cognitivas.""",
        1
    ),

    # =========================================================================
    # CLASE 0: TEXTOS GENUINAMENTE HUMANOS (60 MUESTRAS MULTI-DOMINIO)
    # =========================================================================

    # --- Estudiantes, Escolares y Ensayos Juveniles (Español) ---
    (
        """Ayer fui con mis amigos del colegio al parque a jugar fútbol después de clases. Estuvo lloviendo un poco
        pero igual nos quedamos jugando hasta que anocheció. Mi mamá me retó porque llegué con las zapatillas llenas
        de barro y tuve que lavarlas antes de cenar. Para la próxima tengo que llevar ropa de cambio.""",
        0
    ),
    (
        """Para la clase de historia el profe nos pidió hacer una maqueta sobre las pirámides de Egipto. Mi grupo
        compró cartón prensado y arena en la librería del frente, pero cuando le pusimos la plasticola líquida se ablandó
        toda la base y quedó re torcida. Nos quedamos hasta las dos de la mañana arreglándola con cinta de embalar.
        Al final el profe nos puso un 8 porque le gustó la explicación oral, menos mal.""",
        0
    ),
    (
        """Sinceramente no entiendo a la gente que dice que matemáticas es aburrida. Cuando un ejercicio de álgebra
        no te sale por media hora y de repente ves dónde estaba el error de signo, se siente genial. Es como destrabar
        un nivel difícil en un videojuego. Obvio que cuando no te sale nada querés tirar el cuaderno por la ventana,
        pero cuando le agarrás la mano está buenísimo.""",
        0
    ),
    (
        """En mi colegio hicieron una feria de ciencias la semana pasada y a mi curso le tocó el stand del agua.
        Hicimos un filtro casero con botellas de plástico cortadas, arena fina, carbón activado y piedritas de río.
        Echamos agua con tierra y salía bastante clarita, aunque nadie del jurado se animó a tomar un trago por si acaso.
        Nos reímos un montón con mis compañeros preparando los afiches con cartulinas de colores.""",
        0
    ),
    (
        """Tengo que entregar el ensayo de literatura el viernes y todavía voy por la mitad. Elegí analizar El túnel
        de Ernesto Sabato porque me atrapó desde la primera página la locura de Castel. Es un personaje re oscuro y paranoico,
        pero a la vez te da lástima cómo no puede conectar con nadie sin obsesionarse. Espero que a la profe no le parezca
        demasiado informal la forma en que lo redacté.""",
        0
    ),
    (
        """Me acuerdo clarito del día que adopté a mi perro Firulais. Estaba lloviendo a cántaros y lo encontramos
        temblando abajo de una caja de cartón cerca de la panadería. Estaba flaco y todo sucio. Mi hermana y yo le rogamos
        a mi papá para llevarlo a casa solo por esa noche, y bueno, ya pasaron cuatro años y ahora es el dueño del sillón.""",
        0
    ),

    # --- Anécdotas Cotidianas y Vida Personal (Español) ---
    (
        """Ayer fui al mercado central a buscar fruta y no encontré nada bueno. La verdad es que los precios
        están por las nubes, una papaya me costó casi el doble que la semana pasada. Me encontré con don Carlos,
        el que vende quesos en la esquina, y me estuvo contando que el camión no pudo llegar por los bloqueos
        en la carretera. Menudo dolor de cabeza. Al final me regresé a la casa solo con dos plátanos y medio kilo
        de limones, con rabia de no haber ido más temprano.""",
        0
    ),
    (
        """No me convence para nada el nuevo final de la serie. O sea, ¿después de seis temporadas creando
        tensión entre los protagonistas deciden resolverlo todo en una escena de dos minutos? Es un chiste.
        Entiendo que los guionistas tenían prisa por terminar o que se les acabó el presupuesto, pero dejar
        tantos cabos sueltos me pareció una falta de respeto con los fans que nos bancamos cada domingo religiosamente.""",
        0
    ),
    (
        """Me desperté con el timbre a las seis de la mañana y era el repartidor con una caja pesadísima que ni siquiera
        era para mí. Se había confundido de piso. Le señalé el número en la pared con cara de dormido, me pidió disculpas
        a media voz y se fue arrastrando los pies por el pasillo. Volverme a dormir después de eso fue una misión imposible,
        así que me preparé unos mates y me puse a ordenar el placard.""",
        0
    ),
    (
        """El café de la mañana es sagrado, viejo. Si no me tomo por lo menos una taza bien cargada antes de
        abrir el correo, no funciono. La gente dice que es pura adicción a la cafeína, y probablemente tengan
        razón, pero ese olorcito cuando empieza a colar en la greca no lo cambio por nada en este mundo.
        Hoy se me derramó un poco en la hornilla y olía a quemado por todo el apartamento, qué desastre.""",
        0
    ),
    (
        """Llegué empapado a la oficina porque se largó a llover de golpe justo cuando salí de la estación del metro.
        No llevaba paraguas ni campera impermeable, así que me tocó correr como tres cuadras esquivando charcos.
        Mis medias quedaron hechas una sopa y tuve que pasarme toda la reunión de las diez tratando de disimular
        el frío que tenía en los pies. Odio los días grises así.""",
        0
    ),
    (
        """Mi abuela siempre decía que para que la masa de las empanadas quede crujiente hay que usar grasa de cerdo
        bien fría y amasar lo menos posible. Yo antes no le hacía caso y usaba aceite vegetal común, pero la verdad
        es que la diferencia es del cielo a la tierra. El fin de semana pasado hice una docena de carne picada a cuchillo
        con cebolla de verdeo y comino, y volaron en media hora.""",
        0
    ),
    (
        """Me parece rarísimo que todavía haya gente que use cheques en papel en pleno 2026. Hoy estuve media hora
        haciendo fila en el banco porque un tipo adelante mío tenía una libreta entera para depositar uno por uno.
        El cajero lo miraba con cara de resignación y yo atrás mirando el reloj porque llegaba tarde a buscar a mi hija al colegio.""",
        0
    ),
    (
        """El fin de semana salimos a pedalear por la costanera con dos amigos. Hacía un viento en contra tremendo
        que te frenaba en seco en las subidas. A los diez kilómetros a Juan se le pinchó la rueda trasera; paramos en una
        estación de servicio a parcharla con pegamento viejo que casi no pegaba nada. Tardamos como cuarenta minutos entre
        risas y quejas, pero la cerveza fría que nos tomamos al llegar pagó con creces todo el cansancio.""",
        0
    ),
    (
        """No entiendo por qué complican tanto el trámite para renovar el carnet de conducir. Te piden tres fotos carnet
        con fondo blanco, pero cuando llegás a la oficina te sacan ellos mismos una foto con una webcam de hace diez años.
        Perdés toda la mañana de ventanilla en ventanilla para que te pongan un sello en una hoja que después tiran a un cajón.""",
        0
    ),

    # --- Escritura Humana Profesional, Técnica y Académica (Español) ---
    (
        """El experimento falló en tres ocasiones consecutivas antes de que advirtiéramos la fuga en la junta de teflón.
        Aunque la literatura previa sugiere estabilidad térmica hasta los 180 grados, nuestras lecturas manométricas
        se desplomaron abruptamente. Cambiamos el sellante. Reanudamos la purga con argón. A partir de ese ajuste
        puntual, la reacción transcurrió con total normalidad durante las cuatro horas restantes.""",
        0
    ),
    (
        """Llevo quince años enseñando historia en escuelas públicas y cada vez noto menos interés por memorizar fechas
        y más curiosidad por las vivencias individuales. A los alumnos les da igual qué tratado formal se firmó en 1815;
        quieren saber qué comía un soldado en las trincheras, cuánto cobraba o si le temblaban las manos antes del combate.
        Y tienen toda la razón del mundo. La historia viva está en las cartas personales, no en los decretos secos.""",
        0
    ),
    (
        """Estuvimos discutiendo en el grupo si valía la pena meterle más plata a la campaña de anuncios o si
        era mejor cambiar la imagen del producto. Honestamente creo que la página web carga demasiado lento
        en celulares y la gente se va antes de ver la oferta. Hice una prueba con mi teléfono usando datos
        y tardó casi ocho segundos. Con esa velocidad nadie va a comprar nada, por más plata que le tiremos a Facebook.""",
        0
    ),
    (
        """Al revisar el código del backend notamos que las consultas a la base de datos se ejecutaban de forma síncrona
        dentro de un bucle for. Bastó reemplazar esa sección por una consulta única con cláusula IN para que el tiempo
        de respuesta cayera de novecientos milisegundos a escasos cuarenta. Un error infantil de principiante,
        pero que en producción con mil usuarios simultáneos te tira el servidor abajo sin pedir permiso.""",
        0
    ),
    (
        """Durante el relevamiento de campo en la cuenca del río Salado recolectamos cincuenta muestras de suelo
        a profundidades variables entre diez y treinta centímetros. La salinidad resultó notablemente superior a la esperada
        en los sectores colindantes a la vieja ruta provincial. Las muestras fueron etiquetadas y refrigeradas a cuatro
        grados en conservadoras de telgopor para su posterior remisión al laboratorio químico de la facultad.""",
        0
    ),
    (
        """El paciente masculino de 42 años ingresó a la guardia refiriendo dolor epigástrico punzante de tres horas
        de evolución, acompañado de náuseas sin vómitos. El electrocardiograma basal no evidenció signos de isquemia aguda.
        Se administró analgesia endovenosa y se solicitaron enzimas cardíacas seriadas, las cuales arrojaron valores
        dentro de parámetros normales. Fue dado de alta con pautas de alarma e interconsulta con gastroenterología.""",
        0
    ),
    (
        """En este trabajo nos proponemos indagar cómo influyeron las migraciones internas en la conformación del cordón
        industrial bonaerense entre 1945 y 1955. Lejos de constituir un proceso homogéneo, los testimonios orales de época
        muestran tensiones marcadas entre los obreros radicados con anterioridad y los recién llegados de las provincias
        del norte. Creemos que estas fracturas iniciales explican en parte las disputas sindicales de la década siguiente.""",
        0
    ),

    # --- Opiniones, Blogs y Críticas Culturales (Español) ---
    (
        """Vi la última película de Tarantino en el cine del centro y salí con sensaciones encontradas. Los diálogos
        tienen esa chispa inconfundible de siempre y la banda sonora es una maravilla absoluta, pero sentí que la última
        media hora se estira sin necesidad para llegar a un clímax que ya te veías venir desde el trailer. Igual vale la entrada,
        aunque no esté al nivel de sus clásicos indiscutidos.""",
        0
    ),
    (
        """El partido de anoche fue una montaña rusa de nervios. Arrancamos perdiendo a los cinco minutos por un penal
        regalado que cobró el árbitro sin dudar, pero el equipo reaccionó en el segundo tiempo con garra pura.
        El gol del empate en el minuto 89 desató una locura total en la tribuna. Nos quedamos afónicos de gritar.""",
        0
    ),
    (
        """Me compré unos auriculares inalámbricos de gama media porque los míos con cable se rompieron en el colectivo.
        La cancelación de ruido anda bastante bien en la calle, pero el micrófono para llamadas es malísimo; cada vez
        que llamo a mi vieja me dice que parezco estar hablando desde adentro de una lata de conserva.""",
        0
    ),
    (
        """A veces pienso que las redes sociales nos quemaron la cabeza con la idea de que todo tiene que ser productivo.
        Si te pasás un domingo entero mirando el techo o durmiendo la siesta te sentís culpable, como si estuvieras
        desperdiciando tu vida. Yo reivindico el derecho a no hacer absolutamente nada de vez en cuando sin dar explicaciones.""",
        0
    ),

    # --- Textos Humanos en Inglés (Student Essays, Casual Narratives, Tech Reviews) ---
    (
        """I tried making sourdough bread yesterday and totally messed it up. The starter was bubbling fine,
        but I think my kitchen was just too cold or I didn't knead it enough. The loaf ended up looking like
        a flat, dense brick. It tasted alright with some salted butter, but definitely not what you'd post on Instagram.
        Gonna try again this weekend with warmer water.""",
        0
    ),
    (
        """Honestly wasn't expecting much from the local indie gig on Friday, but those guys actually killed it.
        The drummer was an absolute beast, dropping fills that had everybody in the bar staring. The sound guy
        messed up the levels on the vocals during the first two tracks, but once they dialed it in, the vibe was crazy.
        Glad I didn't stay home watching Netflix.""",
        0
    ),
    (
        """My laptop died in the middle of a client zoom call because my cat unplugged the charger behind the desk.
        I was literally mid-sentence pitching the Q3 budget numbers when the screen went black. Scrambled around
        under the desk in the dark, stubbed my toe on the subwoofer, and by the time I rebooted, they had already
        moved on to the next slide. Brutal morning.""",
        0
    ),
    (
        """Look, if you're gonna hike that trail after heavy rain, bring proper boots. I saw people slipping all
        over the place in regular sneakers, covered in mud up to their knees. The view at the summit is sick,
        don't get me wrong, but scrambling down that wet clay ridge is no joke if you don't have grip.""",
        0
    ),
    (
        """Spent the entire afternoon trying to debug a weird CSS layout issue on our checkout page. Turns out somebody
        set an arbitrary negative margin on a nested div six months ago to patch a mobile bug, and it broke the whole
        grid on desktop Safari. Deleted two lines of hacky CSS and everything snapped right back into place. Felt like magic.""",
        0
    ),
    (
        """For my college history paper I ended up reading through dozens of digitized letters written by coal miners
        in Pennsylvania during the 1920s strike. What struck me most wasn't the political slogans, but how worried they were
        about simple things like whether their kids would have shoes for school that winter. You don't get that raw human
        angle from textbooks.""",
        0
    ),
    (
        """My commute takes almost an hour each way on the commuter rail, so I've gotten really into reading paperback sci-fi.
        Finished Dune last week and started Neuromancer. The prose is so dense and punchy that if you miss a stop you don't even care.
        Way better than mindlessly scrolling Twitter until your battery drains to twenty percent.""",
        0
    ),
    (
        """We decided to paint the living room ourselves over the weekend to save some cash. Big mistake. Took us three trips
        to Home Depot just to get the right roller covers and painter's tape, and we still managed to get droplets of off-white
        paint all over the hardwood floor. At least the walls look decent now, as long as you don't inspect the corners too closely.""",
        0
    ),
    (
        """Para la clase de lengua nos hicieron leer los primeros diez capítulos del Quijote. Al principio me costó un montón
        engancharme por el vocabulario antiguo, pero cuando el tipo sale con su armadura toda oxidada pensando que unas
        molinos de viento son gigantes de cuatro brazos, me maté de la risa. Sancho Panza es el mejor personaje lejos,
        porque es el único con dos dedos de frente que trata de hacer entrar en razón a su amo sin que lo mande a pasear.""",
        0
    ),
    (
        """Hicimos una observación con microscopio óptico en el laboratorio del cole. Tuvimos que sacarle una telita
        transparente a una cebolla con una pinza de depilar y ponerle una gota de azul de metileno encima. Se veían clarito
        las paredes celulares como si fueran ladrillitos de una pared. Mi compañero de banco casi rompe el cubreobjetos
        porque bajó el tubo con el tornillo macrométrico con demasiada fuerza, pero por suerte la profe no lo vio.""",
        0
    ),
    (
        """Ayer intenté cocinar arroz con pollo para mi familia y fue un desastre total. Se me quemó el fondo de la olla
        porque me quedé contestando mensajes de WhatsApp y me olvidé de bajar el fuego. Quedó con un gusto a ahumado
        horrible que no se iba ni con un kilo de queso rallado. Terminamos pidiendo empanadas por la aplicación
        y mi hermano todavía se sigue burlando de mis dotes de chef.""",
        0
    ),
    (
        """Tomarse el colectivo 60 en hora pico es para valientes. Venía tan apretado que no podía ni sacar el celular del bolsillo.
        Para colmo el chofer clavó los frenos en la esquina de Callao y salimos todos despedidos hacia adelante como fichas
        de dominó. Una señora me pisó sin querer y me clavó el taco en el empeine; casi me pongo a llorar del dolor
        pero tuve que poner cara de póker para no pasar vergüenza delante de todo el mundo.""",
        0
    ),
    (
        """Nos quedamos jugando partidas clasificatorias con mis amigos hasta las tres de la madrugada. Estábamos a una sola
        victoria de subir de rango y en la última ronda a Martín se le cayó la conexión a internet de la nada.
        Nos pasamos veinte minutos puteando por Discord mientras el equipo rival nos ganaba la partida con uno menos.
        Nos fuimos a dormir con una bronca tremenda jurando no volver a jugar nunca más, aunque seguro hoy a la noche volvemos.""",
        0
    ),
    (
        """Entramos al aula confiados y el profe de física nos tiró un examen sorpresa de tiro oblicuo sin previo aviso.
        Nadie se acordaba de las fórmulas de memoria porque habíamos estudiado para la entrega de historia del día siguiente.
        Me pasé la hora y media despejando variables al voleo y rezando para que los números me dieran algo con sentido físico.
        Cuando salimos al pasillo a comparar resultados con los chicos, a cada uno le había dado un valor completamente distinto.""",
        0
    ),
    (
        """Cada vez que hay tormenta eléctrica mi gata Michi se mete abajo del placard y no sale por nada del mundo.
        Le dejamos su manta favorita y un platito con atún cerca de la puerta, pero no hay caso; prefiere quedarse
        hecha un ovillo en el rincón más oscuro hasta que pasa el último trueno. Recién a la mañana siguiente aparece
        ronroneando y pidiendo mimos como si no hubiera pasado nada.""",
        0
    ),
    (
        """Fuimos al recital de rock en el estadio y la acústica en la parte de campo trasero era bastante floja.
        El bajo tapaba casi por completo la voz del cantante durante la primera mitad del show, pero la energía de la gente
        hizo que no importara nada. Armaron un pogo gigante en el medio de la pista y casi pierdo la billetera del bolsillo.
        Valió cada centavo de la entrada.""",
        0
    ),
    (
        """En mi barrio tenemos un problema grave con el arroyo que pasa a tres cuadras de mi casa. La gente tira bolsas
        de basura en la orilla y cuando llueve fuerte el agua sube y se inunda toda la esquina. Con un grupo de vecinos
        organizamos una jornada de limpieza el sábado pasado y juntamos más de veinte bolsas de consorcio llenas de botellas
        y neumáticos viejos. Es una gota en el océano pero por algo hay que empezar si queremos que el municipio nos dé bola.""",
        0
    ),
    (
        """Mi hermano chico se pasa el día entero mirando videos cortos en TikTok con el volumen al mango. Le decís que baje
        el sonido porque estás estudiando o trabajando y te mira con cara de que no entiende nada. A veces me da un poco
        de miedo ver cómo los pibes de ahora tienen una capacidad de atención de diez segundos; no pueden sentarse a mirar
        una película entera de corrido sin revisar el teléfono cada cinco minutos.""",
        0
    ),
    (
        """I accidentally washed my favorite wool sweater in hot water and now it literally looks like it would fit a toddler.
        I tried that trick where you soak it in hair conditioner and gently stretch the fibers out on a towel, but it just
        ended up misshapen and weirdly stiff around the armpits. Into the donation bin it goes, I guess. Lesson learned.""",
        0
    ),
    (
        """Pulling an all-nighter in the college library during finals week is basically a collective fever dream.
        By 4 AM the vending machines were completely out of energy drinks, so people were taking turns making instant espresso
        with hot tap water in the bathroom sinks. Somebody fell asleep face-down on their open textbook and drooled across
        three pages of organic chemistry diagrams.""",
        0
    ),
    (
        """Tried my hand at film photography with an old Pentax camera I dug out of my uncle's attic. Took 36 shots over
        the course of two weeks and dropped the roll off at the local lab. Half of them came out completely underexposed
        because the light meter battery was dead, but the four shots that actually worked have this gorgeous grainy warmth
        that no phone filter can replicate.""",
        0
    ),
    (
        """Our road trip through the desert almost got ruined when the AC in the rental car started blowing lukewarm air
        right around noon. Outside temperature was pushing 104 degrees. We rolled the windows down, but that just felt like
        sticking our heads directly into an oven. Stopped at a tiny gas station, bought a giant bag of crushed ice,
        and took turns holding ice cubes against our foreheads until we hit the state line.""",
        0
    ),
    (
        """Honestly sick of every single household appliance needing a Wi-Fi connection and an app. My new microwave literally
        refused to heat a bowl of soup until I agreed to its terms of service update on my phone. Why does a box that shoots
        microwaves at leftover pizza need to collect telemetry data on my heating habits? It's absurd.""",
        0
    )
]

def train_and_save_model():
    """Entrena la red neuronal y ensamble tripartito y guarda el modelo serializado."""
    print("Iniciando extracción de 30 características forenses en dataset ampliado...")
    X = []
    y = []

    for text, label in TRAINING_DATA:
        feats = extract_features(text)
        X.append(feats["features_vector"])
        y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    n_ai = int(sum(y == 1))
    n_human = int(sum(y == 0))
    print(f"Dataset compilado: {len(X)} muestras (IA: {n_ai}, Humano: {n_human}) con {X.shape[1]} características.")

    # 1. Red Neuronal MLP Profunda
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation='relu',
        solver='adam',
        max_iter=2500,
        random_state=42,
        alpha=0.005,
        learning_rate_init=0.006,
        early_stopping=False
    )

    # 2. Gradient Boosting Classifier
    gb = GradientBoostingClassifier(
        n_estimators=90,
        learning_rate=0.07,
        max_depth=4,
        subsample=0.85,
        random_state=42
    )

    # 3. Random Forest Robusto
    rf = RandomForestClassifier(
        n_estimators=90,
        max_depth=7,
        random_state=42
    )

    # Ensamble de votación probabilística suave (Soft Voting)
    ensemble = VotingClassifier(
        estimators=[
            ('neural_net', mlp),
            ('gradient_boost', gb),
            ('random_forest', rf)
        ],
        voting='soft'
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', ensemble)
    ])

    print("Entrenando ensamble multicapa...")
    pipeline.fit(X, y)

    score = pipeline.score(X, y)
    print(f"Precisión en conjunto de entrenamiento: {score * 100:.2f}%")

    output_path = os.path.join(os.path.dirname(__file__), "model_weights.joblib")
    joblib.dump(pipeline, output_path)
    print(f"Modelo forense guardado en: {output_path}")

    return pipeline

if __name__ == "__main__":
    train_and_save_model()
