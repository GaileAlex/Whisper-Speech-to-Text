FROM eclipse-temurin:25-jdk-noble AS builder
WORKDIR /workspace
COPY .mvn/ .mvn
COPY mvnw pom.xml ./
RUN ./mvnw dependency:go-offline
COPY src ./src
RUN ./mvnw package -DskipTests

FROM eclipse-temurin:25-jre-noble

WORKDIR /app
COPY --from=builder /workspace/target/*.jar app.jar

RUN useradd --create-home spring
USER spring

EXPOSE 8389
# Netty loads its native epoll transport: without the flag Java 25 warns that this will be blocked
ENTRYPOINT ["java", "--enable-native-access=ALL-UNNAMED", "-jar", "app.jar"]
