#!/usr/bin/env groovy
@Library('com.jenkins.pipeline.library@v0.3.16') _

// Define variable to a scope of this file
def glTwistlockScanResult

def acr_env_map = [dev: "xx", prod: "xx"]
def sp_map = [dev: "oco_ds_dev_sp", prod: "oco_ds_sp"]

def img_map = [inference: ".inference", train: ""]
def imgnm_map = [inference: "_inference", train: ""]
def ACR_CREDENTIALS
def DOCKER_FILE
def DOCKER_IMAGE
def ACR_URL

pipeline {
    agent {
        label 'docker-kitchensink-slave'
    }
    parameters{

        choice(name: 'ENVIRONMENT', choices: ['prod', 'dev'] )
        choice(name: 'IMG_TYPE', choices: ['train', 'inference'] )

    }
    environment {
        DOCKER_IMAGE_TAG = 'latest'
        TWISTLOCK_CREDS = 'prisma_login'
        VULNERABILITY_THRESHOLD = 'high'
        COMPLIANCE_THRESHOLD = 'high'
        FAIL_BUILD = false
        EMAIL_ADDR = 'xxx@xxx.com'
    }

    stages {
        stage ('Docker Build') {
            steps {
                script {
                    ACR_CREDENTIALS = "${sp_map[params.ENVIRONMENT]}"
                    DOCKER_FILE = "Dockerfile${img_map[params.IMG_TYPE]}"
                    DOCKER_IMAGE = "sdkv2_agent_img${imgnm_map[params.IMG_TYPE]}"
                    ACR_URL = "${acr_env_map[params.ENVIRONMENT]}.azurecr.io"
                    ENV = "${params.ENVIRONMENT}"
                }
                // first stage base azureml image build
                sh """
                    docker build --no-cache --pull -t ${DOCKER_IMAGE}:${env.DOCKER_IMAGE_TAG} . -f ${DOCKER_FILE}
                    docker tag ${DOCKER_IMAGE}:${env.DOCKER_IMAGE_TAG} ${ACR_URL}/${DOCKER_IMAGE}:${env.DOCKER_IMAGE_TAG}
                """
            }
        }
        stage ('Docker Push') {
            steps {
                withCredentials([azureServicePrincipal("${ACR_CREDENTIALS}")]) {
                    // login with non-user ID
                    sh """
                        docker login ${ACR_URL} --username="${AZURE_CLIENT_ID}" --password="${AZURE_CLIENT_SECRET}"
                        docker push ${ACR_URL}/${DOCKER_IMAGE}:${env.DOCKER_IMAGE_TAG}
                    """
                }
            }
        }
    }

    post {
        always {
            echo 'This will always run'
            emailext body: """
                         Build URL: ${BUILD_URL} <br>
                         TwistlockScanResult: ${glTwistlockScanResult} <br>
                         TwistlockScan: https://containersecurity.com/#!/monitor/vulnerabilities/images/ci?search=${DOCKER_IMAGE}:${env.DOCKER_IMAGE_TAG} <br>
                         The tag '${env.DOCKER_IMAGE_TAG}' for the container ${DOCKER_IMAGE} has a new image
                     """,
                     mimeType: 'text/html',
                     subject: "$currentBuild.currentResult-$JOB_NAME",
                     to: "${env.EMAIL_ADDR}"
        }
        success {
            echo 'This will run only if successful'
        }
        failure {
            echo 'This will run only if failed'
        }
        unstable {
            echo 'This will run only if the run was marked as unstable'
        }
        changed {
            echo 'This will run only if the state of the Pipeline has changed'
            echo 'For example, if the Pipeline was previously failing but is now successful'
        }
    }
}